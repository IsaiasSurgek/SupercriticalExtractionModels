"""Estimação de parâmetros com multi-start e diagnóstico de identificabilidade.

Objetivo minimizado: resíduos **relativos** ``(R_exp - R_calc)/R_exp``, via
``scipy.optimize.least_squares``. Métricas reportadas: DRM (média de
|resíduo relativo| x 100) e R² convencional.

Duas decisões que não são cosméticas
------------------------------------
**Multi-start é obrigatório.** Uma partida única cai em mínimos locais
diferentes conforme a condição, e o resultado passa a depender do chute. Aqui
as partidas vêm de hipercubo latino, em escala log para os coeficientes.

**Bounds são físicos, não numéricos.** Nada de ``1e5`` para conter o
otimizador: se um parâmetro encosta num limite, isso é diagnóstico e é
reportado, não maquiado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

# --- limites físicos ---------------------------------------------------------
# a0 = 6(1-eps)/d_p ~ 2e3 m2/m3;  k_f ~ 1e-6 a 1e-2 m/s  =>  Kfa0 ~ 2e-3 a 2e1
KFA0_MIN, KFA0_MAX = 1e-6, 1e2
# transporte intraparticular e' ordens de grandeza mais lento
KSA0_MIN, KSA0_MAX = 1e-9, 1e0
SA_MIN, SA_MAX = 1e-6, 1.0 - 1e-6
YR_MIN, YR_MAX = 1e-5, 60e-3          # kg/kg  (0,01 a 60 g/kg)
B_MIN, B_MAX = 1.0, 1e7               # s      (1 s a 116 dias)
E_LIM_MIN = 1e-6                      # kg/kg; o teto vem de x0

Transformacao = Literal["linear", "log10", "logit10"]


@dataclass(frozen=True)
class Param:
    """Descrição de um parâmetro estimável.

    ``transformacao`` define o espaço em que o otimizador trabalha e em que as
    partidas são sorteadas:

    * ``"log10"`` para coeficientes que varrem ordens de grandeza;
    * ``"logit10"`` para frações em (0,1) cujos dois extremos importam. Sortear
      ``sa`` uniformemente parece razoável e não é: há condições que só ajustam
      com ``sa > 0,98``, uma fatia de 2 % que partidas uniformes quase nunca
      alcançam, e o otimizador "converge" para um mínimo local ruim.
    """

    nome: str
    lo: float
    hi: float
    transformacao: Transformacao
    unidade: str

    def para_interno(self, v):
        if self.transformacao == "log10":
            return np.log10(v)
        if self.transformacao == "logit10":
            v = np.clip(v, 1e-12, 1 - 1e-12)
            return np.log10(v / (1 - v))
        return v

    def para_natural(self, x):
        if self.transformacao == "log10":
            return 10.0**x
        if self.transformacao == "logit10":
            return 1.0 / (1.0 + 10.0 ** (-x))
        return x

    @property
    def lo_interno(self) -> float:
        return float(self.para_interno(self.lo))

    @property
    def hi_interno(self) -> float:
        return float(self.para_interno(self.hi))


@dataclass
class Ajuste:
    """Resultado de um ajuste multi-start.

    Attributes
    ----------
    params : list[Param]
    theta : np.ndarray
        Melhor solução, em unidades naturais.
    drm : float
        Desvio relativo médio [%].
    r2 : float
    n_minimos : int
        Mínimos locais distintos encontrados.
    dispersao : dict[str, tuple[float, float]]
        Faixa [min, max] de cada parâmetro entre as soluções quase-ótimas (DRM
        até 1 % acima do melhor). Faixa larga = superfície plana = o valor
        pontual é escolha do otimizador, não informação dos dados.
    nos_limites : list[str]
        Parâmetros que encostaram num bound.
    """

    params: list[Param]
    theta: np.ndarray
    drm: float
    r2: float
    n_starts: int = 0
    n_convergidos: int = 0
    n_minimos: int = 1
    dispersao: dict[str, tuple[float, float]] = field(default_factory=dict)
    nos_limites: list[str] = field(default_factory=list)

    def valor(self, nome: str) -> float:
        for i, p in enumerate(self.params):
            if p.nome == nome:
                return float(self.theta[i])
        raise KeyError(f"parâmetro {nome!r} não está neste ajuste")

    def mal_determinados(self, decadas: float = 2.0) -> list[str]:
        """Parâmetros cuja dispersão passa de ``decadas`` ordens de grandeza."""
        fora = []
        for nome, (lo, hi) in self.dispersao.items():
            if lo > 0 and np.log10(hi / lo) > decadas:
                fora.append(nome)
        return fora


def drm_de(residuo: np.ndarray) -> float:
    """DRM [%] = média(|resíduo relativo|) * 100."""
    return float(np.mean(np.abs(residuo)) * 100.0)


def r2_de(y_exp: np.ndarray, y_calc: np.ndarray) -> float:
    """R² convencional. Invariante a fração vs porcentagem."""
    ss_res = float(np.sum((y_exp - y_calc) ** 2))
    ss_tot = float(np.sum((y_exp - np.mean(y_exp)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def ajustar(
    residuo: Callable[[np.ndarray], np.ndarray],
    params: Sequence[Param],
    *,
    n_starts: int = 64,
    seed: int = 0,
    fixos: dict[str, float] | None = None,
    sementes: Sequence[np.ndarray] | None = None,
    y_exp: np.ndarray | None = None,
    prever: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Ajuste:
    """Ajusta por mínimos quadrados com multi-start por hipercubo latino.

    Parameters
    ----------
    residuo : callable
        Recebe o vetor completo de parâmetros em unidade natural, devolve o
        vetor de resíduos relativos.
    params : sequence of Param
    n_starts : int
        Partidas do hipercubo latino. Mínimo recomendado: 50.
    fixos : dict, optional
        ``{nome: valor}`` de parâmetros congelados (usado no perfil).
    sementes : sequence of ndarray, optional
        Partidas adicionais em unidade natural.
    y_exp, prever : optional
        Se dados, calcula R².
    """
    params = list(params)
    fixos = dict(fixos or {})
    livres = [i for i, p in enumerate(params) if p.nome not in fixos]
    if not livres:
        raise ValueError("todos os parâmetros estão fixos: nada a ajustar")

    base = np.array(
        [fixos.get(p.nome, 0.5 * (p.lo + p.hi)) for p in params], dtype=float
    )
    lo = np.array([params[i].lo_interno for i in livres])
    hi = np.array([params[i].hi_interno for i in livres])

    def montar(x: np.ndarray) -> np.ndarray:
        theta = base.copy()
        for k, i in enumerate(livres):
            theta[i] = params[i].para_natural(x[k])
        return theta

    def fun(x: np.ndarray) -> np.ndarray:
        r = residuo(montar(x))
        return np.full_like(r, 1e3) if not np.all(np.isfinite(r)) else r

    partidas: list[np.ndarray] = []
    if n_starts > 0:
        amostra = qmc.LatinHypercube(d=len(livres), seed=seed).random(n_starts)
        partidas = list(lo + amostra * (hi - lo))
    for s in sementes or []:
        s = np.asarray(s, dtype=float)
        partidas.append(
            np.clip(np.array([params[i].para_interno(s[i]) for i in livres]), lo, hi)
        )
    if not partidas:
        raise ValueError("nenhuma partida")

    solucoes: list[tuple[float, np.ndarray]] = []
    for x0 in partidas:
        try:
            r = least_squares(fun, x0, bounds=(lo, hi), method="trf")
        except (ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(r.fun)):
            solucoes.append((drm_de(r.fun), montar(r.x)))
    if not solucoes:
        raise RuntimeError("nenhuma partida convergiu")
    solucoes.sort(key=lambda z: z[0])

    melhor_drm, melhor = solucoes[0]

    # mínimos locais distintos: DRM e posição ambos diferentes
    def xint(th):
        return np.array([params[i].para_interno(th[i]) for i in livres])

    reps: list[tuple[float, np.ndarray]] = []
    for d, th in solucoes:
        if all(
            abs(d - dr) > 1e-3 or np.max(np.abs(xint(th) - xint(tr))) > 0.05
            for dr, tr in reps
        ):
            reps.append((d, th))

    quase = np.array([th for d, th in solucoes if d <= melhor_drm * 1.01])
    dispersao = {
        p.nome: (float(quase[:, i].min()), float(quase[:, i].max()))
        for i, p in enumerate(params)
        if p.nome not in fixos
    }

    nos_limites = []
    for i, p in enumerate(params):
        if p.nome in fixos:
            continue
        largura = p.hi_interno - p.lo_interno
        xi = float(p.para_interno(melhor[i]))
        if xi - p.lo_interno < 1e-4 * largura:
            nos_limites.append(f"{p.nome} no limite inferior ({p.lo:g} {p.unidade})")
        elif p.hi_interno - xi < 1e-4 * largura:
            nos_limites.append(f"{p.nome} no limite superior ({p.hi:g} {p.unidade})")

    r2 = float("nan")
    if y_exp is not None and prever is not None:
        r2 = r2_de(y_exp, prever(melhor))

    return Ajuste(
        params=params,
        theta=melhor,
        drm=melhor_drm,
        r2=r2,
        n_starts=len(partidas),
        n_convergidos=len(solucoes),
        n_minimos=len(reps),
        dispersao=dispersao,
        nos_limites=nos_limites,
    )


# =============================================================================
# Perfil de verossimilhança
# =============================================================================


@dataclass
class Perfil:
    """Perfil de verossimilhança de um parâmetro.

    Attributes
    ----------
    grade : np.ndarray
        Valores fixados, em unidade natural.
    drm : np.ndarray
        DRM mínimo [%] após reotimizar todos os demais em cada ponto.
    status : str
        ``"identificavel"``, ``"limite_inferior"``, ``"limite_superior"`` ou
        ``"nao_identificavel"``.
    faixa : tuple[float, float]
        Intervalo onde o DRM fica abaixo do limiar.
    """

    nome: str
    param: Param
    grade: np.ndarray
    drm: np.ndarray
    drm_min: float
    valor_otimo: float
    status: str
    faixa: tuple[float, float]
    limiar: float

    @property
    def largura_decadas(self) -> float:
        lo, hi = self.faixa
        return float(self.param.para_interno(hi) - self.param.para_interno(lo))

    def texto(self) -> str:
        """Como reportar. Sem valor pontual quando não identificável."""
        lo, hi = self.faixa
        if self.status == "identificavel":
            return f"{self.valor_otimo:.4g}  [{lo:.4g}; {hi:.4g}]"
        if self.status == "limite_inferior":
            return f"> {lo:.4g}"
        if self.status == "limite_superior":
            return f"< {hi:.4g}"
        return f"nao determinado ({lo:.4g} a {hi:.4g})"


def perfilar(
    residuo: Callable[[np.ndarray], np.ndarray],
    params: Sequence[Param],
    theta_otimo: np.ndarray,
    *,
    nomes: Sequence[str] | None = None,
    n_grade: int = 25,
    n_starts: int = 4,
    seed: int = 0,
    limiar_fator: float = 1.10,
    piso_pp: float = 0.05,
    max_decadas: float = 2.0,
) -> dict[str, Perfil]:
    """Perfil de verossimilhança de cada parâmetro.

    Para cada parâmetro: fixa o valor numa grade que cobre os bounds inteiros,
    **reotimiza todos os demais**, e anota o DRM mínimo. Reotimizar é o que
    distingue um perfil de um corte da superfície — sem isso, a correlação
    entre parâmetros aparece como falsa precisão.

    Critérios de "identificável", os três juntos:

    1. o DRM sobe acima de ``DRM_min * limiar_fator + piso_pp`` dos dois lados;
    2. a faixa resultante é mais estreita que ``max_decadas``;
    3. o ótimo global cai dentro dela.

    O piso absoluto existe porque o critério relativo degenera quando o DRM
    mínimo é pequeno. Ele é físico: separar dois ajustes por menos de 0,05 pp
    é separar ruído de pesagem.
    """
    params = list(params)
    idx = {p.nome: i for i, p in enumerate(params)}
    alvos = list(nomes) if nomes is not None else [p.nome for p in params]
    perfis: dict[str, Perfil] = {}

    for nome in alvos:
        p = params[idx[nome]]
        x_ot = float(np.clip(p.para_interno(theta_otimo[idx[nome]]),
                             p.lo_interno, p.hi_interno))
        # grade grossa sobre os bounds + refino em torno do ótimo: com passo
        # grosso o mínimo cai entre nós e a faixa reportada encolhe sem motivo
        grossa = np.linspace(p.lo_interno, p.hi_interno, n_grade)
        fina = np.linspace(max(p.lo_interno, x_ot - 1), min(p.hi_interno, x_ot + 1),
                           max(n_grade // 2, 5))
        xs = np.unique(np.concatenate([grossa, fina, [x_ot]]))
        grade = np.asarray(p.para_natural(xs), dtype=float)

        drms = np.full(grade.size, np.inf)
        thetas: list[np.ndarray | None] = [None] * grade.size
        i_ot = int(np.argmin(np.abs(grade - theta_otimo[idx[nome]])))

        def no(k: int, sementes: list[np.ndarray], n_st: int) -> None:
            try:
                a = ajustar(residuo, params, n_starts=n_st, seed=seed + k,
                            fixos={nome: float(grade[k])}, sementes=sementes)
            except (RuntimeError, ValueError):
                return
            if a.drm < drms[k]:
                drms[k] = a.drm
                thetas[k] = a.theta

        # continuação: cada nó semeia o seguinte, partindo do ótimo para fora.
        # Sem isso nós isolados param em mínimos locais e viram picos que
        # quebram a região contígua abaixo do limiar.
        for sentido in (+1, -1):
            anterior = theta_otimo
            k = i_ot if sentido == +1 else i_ot - 1
            while 0 <= k < grade.size:
                no(k, [anterior, theta_otimo], n_starts)
                if thetas[k] is not None:
                    anterior = thetas[k]
                k += sentido

        # refinamento: reotimiza nós acima de um vizinho, aceitando só melhora
        for _ in range(3):
            melhorou = False
            for k in range(grade.size):
                viz = [thetas[j] for j in (k - 1, k + 1)
                       if 0 <= j < grade.size and thetas[j] is not None]
                if not viz or all(
                    drms[k] <= drms[j] + 1e-9
                    for j in (k - 1, k + 1) if 0 <= j < grade.size
                ):
                    continue
                antes = drms[k]
                no(k, viz, 0)
                melhorou |= drms[k] < antes - 1e-9
            if not melhorou:
                break

        fin = np.isfinite(drms)
        if not fin.any():
            continue
        drm_min = float(np.min(drms[fin]))
        limiar = drm_min * limiar_fator + piso_pp
        status, faixa = _classificar(grade, drms, limiar, p, max_decadas)

        v_ot = float(theta_otimo[idx[nome]])
        if not (faixa[0] <= v_ot <= faixa[1]):
            faixa = (min(faixa[0], v_ot), max(faixa[1], v_ot))
            status = "nao_identificavel"

        perfis[nome] = Perfil(nome, p, grade, drms, drm_min, v_ot,
                              status, faixa, limiar)
    return perfis


def _classificar(grade, drm, limiar, param, max_decadas):
    """Classifica o perfil pela região contígua abaixo do limiar."""
    if not np.any(drm <= limiar):
        return "nao_identificavel", (float(grade[0]), float(grade[-1]))

    i = int(np.argmin(drm))
    lo_i = hi_i = i
    while lo_i - 1 >= 0 and drm[lo_i - 1] <= limiar:
        lo_i -= 1
    while hi_i + 1 < grade.size and drm[hi_i + 1] <= limiar:
        hi_i += 1

    toca_esq, toca_dir = lo_i == 0, hi_i == grade.size - 1

    def cruzamento(dentro, fora):
        y0, y1 = drm[dentro], drm[fora]
        if not np.isfinite(y1) or y1 <= y0:
            return float(grade[dentro])
        x0 = float(param.para_interno(grade[dentro]))
        x1 = float(param.para_interno(grade[fora]))
        return float(param.para_natural(x0 + (limiar - y0) * (x1 - x0) / (y1 - y0)))

    faixa = (
        float(grade[0]) if toca_esq else cruzamento(lo_i, lo_i - 1),
        float(grade[-1]) if toca_dir else cruzamento(hi_i, hi_i + 1),
    )

    if toca_esq and toca_dir:
        return "nao_identificavel", faixa
    if toca_esq:
        return "limite_superior", faixa
    if toca_dir:
        return "limite_inferior", faixa

    # mínimo interior não basta: faixa larga demais não serve para comparar
    # condições nem correlacionar com pressão e temperatura
    larg = float(param.para_interno(faixa[1]) - param.para_interno(faixa[0]))
    if larg > max_decadas:
        return "nao_identificavel", faixa
    return "identificavel", faixa
