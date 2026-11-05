"""EXECUTE ESTE ARQUIVO.

    python main.py                      tudo: cinetica + solubilidade
    python main.py --perfis             + perfil de verossimilhanca
    python main.py --sem-graficos       so numeros, sem PNG
    python main.py --sem-solubilidade   so Sovova e Esquivel

O que sai de uma execucao completa:

* **Sovova (1994)** e **Esquivel et al. (1999)** ajustados a cada curva, com
  diagnostico de identificabilidade e periodo de Sovova coberto pelos dados;
* **ajuste global** com um unico ``sa`` para todas as curvas;
* **Chrastil (1982)** e **del Valle & Aguilera (1988)** para a solubilidade,
  com teste da premissa de saturacao;
* graficos e CSVs de tudo, na pasta de ``PASTA_SAIDA``.

Edite os dados em `entrada.py`. Este arquivo nao precisa ser alterado.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import entrada as ent
from co2 import densidade_co2, descrever_fonte
from ajuste import (
    B_MIN, B_MAX, E_LIM_MIN, KFA0_MAX, KFA0_MIN, KSA0_MAX, KSA0_MIN,
    SA_MAX, SA_MIN, YR_MAX, YR_MIN,
    Ajuste, Param, ajustar, drm_de, perfilar, r2_de,
)
from modelos import (
    descrever_periodos, esquivel_degenerou, esquivel_e, grupos_sovova, sovova_e,
)

S_POR_MIN = 60.0
G_POR_KG = 1000.0


# =============================================================================
# Preparo dos dados
# =============================================================================


@dataclass(frozen=True)
class Curva:
    """Uma corrida, convertida para SI e com o leito resolvido.

    Attributes
    ----------
    t_s : np.ndarray
        Tempos de coleta [s].
    e_exp : np.ndarray
        Rendimento acumulado [kg óleo / kg inerte], base livre de soluto.
    N_kg : float
        Carga de sólido inerte = massa_seca * (1 - teor de óleo) [kg].
    eps : float
        Porosidade do leito [-].
    Q_kg_s : float
        Vazão de CO2 [kg/s].
    x0 : float
        Teor inicial [kg óleo / kg inerte] = teor/(1 - teor).
    """

    nome: str
    P_bar: float
    T_C: float
    massa_seca_kg: float
    t_s: np.ndarray
    e_exp: np.ndarray
    N_kg: float
    eps: float
    Q_kg_s: float
    rho_co2: float
    rho_solido: float
    x0: float

    @property
    def rotulo(self) -> str:
        return f"{self.nome} ({self.P_bar:.0f} bar / {self.T_C:.0f} C)"

    @property
    def t_min(self) -> np.ndarray:
        return self.t_s / S_POR_MIN

    @property
    def R_pct(self) -> np.ndarray:
        return 100.0 * self.e_exp

    @property
    def q(self) -> np.ndarray:
        """q = (Q/N)*t [kg CO2 / kg inerte]."""
        return (self.Q_kg_s / self.N_kg) * self.t_s

    def q_de(self, t_s: np.ndarray) -> np.ndarray:
        return (self.Q_kg_s / self.N_kg) * np.asarray(t_s, dtype=float)

    @property
    def yr_primeiro_ponto(self) -> float:
        """y_r = m1/(Q*t1) [kg/kg] — convenção do código antigo."""
        return (self.e_exp[0] * self.N_kg) / (self.Q_kg_s * self.t_s[0])

    @property
    def recuperado(self) -> float:
        """Fração do extraível que efetivamente saiu [-]."""
        return float(self.e_exp[-1] / self.x0)


def pasta_saida() -> Path:
    """Pasta onde tudo é gravado, sempre dentro da pasta da ferramenta.

    ``PASTA_SAIDA`` em `entrada.py` é resolvida **relativa a este arquivo**, e
    não ao diretório de onde você chamou o script. Sem isso, rodar
    ``python ferramenta/main.py`` de fora espalharia os resultados no diretório
    corrente em vez de deixá-los junto do código.

    Caminho absoluto em ``PASTA_SAIDA`` continua funcionando e é respeitado
    como está, para quem quiser gravar em outro lugar.
    """
    p = Path(ent.PASTA_SAIDA)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def vazao_g_min(massa_seca_g: float) -> float:
    """Vazão mássica de CO2 [g/min], na ordem de prioridade de `entrada.py`.

    1. ``VAZAO_G_MIN``, se informada — já é massa.
    2. ``VAZAO_BOLHOMETRO_ML_MIN`` convertida pela densidade do CO2 **gasoso**
       no ambiente. O bolhômetro mede gás despressurizado; usar a densidade
       supercrítica aqui erraria por duas a três ordens de grandeza.
    3. ``VAZAO_ESPECIFICA_G_G_MIN * massa_seca``, último recurso.
    """
    if getattr(ent, "VAZAO_G_MIN", None) is not None:
        return float(ent.VAZAO_G_MIN)

    ml_min = getattr(ent, "VAZAO_BOLHOMETRO_ML_MIN", None)
    if ml_min is not None:
        T = getattr(ent, "T_AMBIENTE_C", 25.0)
        rho_gas_g_L = 44.01 / (0.0820574 * (T + 273.15))   # gas ideal, 1 atm
        return float(ml_min) * rho_gas_g_L * 1e-3

    return ent.VAZAO_ESPECIFICA_G_G_MIN * massa_seca_g


def preparar() -> dict[str, Curva]:
    """Lê `entrada.py`, valida e converte tudo para SI."""
    teor = ent.TEOR_OLEO
    if not 0 < teor < 1:
        sys.exit(f"ERRO: TEOR_OLEO deve estar entre 0 e 1 (esta {teor}).")
    x0 = teor / (1.0 - teor)

    v_extrator = ent.VOLUME_EXTRATOR_ML * 1e-6
    v_inertes = ent.VOLUME_INERTES_ML * 1e-6

    curvas: dict[str, Curva] = {}
    for nome, d in ent.CURVAS.items():
        t = np.asarray(d["tempo_min"], dtype=float)
        m = np.asarray(d["massa_g"], dtype=float)
        if t.size != m.size:
            sys.exit(f"ERRO em {nome}: tempo_min tem {t.size} pontos e "
                     f"massa_g tem {m.size}.")
        if t.size < 3:
            sys.exit(f"ERRO em {nome}: sao {t.size} pontos; o modelo tem 3 a 4 "
                     f"parametros e precisa de mais dados que isso.")

        if not ent.MASSA_ACUMULADA:
            m = np.cumsum(m)
        if np.any(np.diff(m) < 0):
            print(f"  [!] {nome}: massa acumulada decresce em algum ponto. "
                  f"Confira se MASSA_ACUMULADA={ent.MASSA_ACUMULADA} esta certo.")
        if np.any(m <= 0):
            sys.exit(f"ERRO em {nome}: ha massa zero ou negativa. O residuo e "
                     f"relativo (divide por R_exp) e nao admite zero.")

        massa_seca_kg = d["massa_seca_g"] * 1e-3
        N_kg = massa_seca_kg * (1.0 - teor)
        e_exp = (m * 1e-3) / N_kg          # kg óleo / kg inerte

        if e_exp[-1] > x0:
            print(f"  [!] {nome}: recuperou {100 * e_exp[-1] / x0:.0f} % do "
                  f"extraivel — acima de 100 %. Confira TEOR_OLEO e massa_seca_g.")

        v_solido = massa_seca_kg / ent.RHO_PARTICULA
        eps = (v_extrator - v_solido - v_inertes) / v_extrator
        if not 0 < eps < 1:
            sys.exit(f"ERRO em {nome}: porosidade {eps:.3f} fora de (0,1). "
                     f"Confira VOLUME_EXTRATOR_ML, RHO_PARTICULA e massa_seca_g.")

        rho_co2, _origem = densidade_co2(
            d["P_bar"], d["T_C"], ent.DENSIDADE_CO2,
            getattr(ent, "FONTE_DENSIDADE", "auto"),
        )

        Q = vazao_g_min(d["massa_seca_g"])
        Q_kg_s = Q / (G_POR_KG * S_POR_MIN)

        curvas[nome] = Curva(
            nome=nome, P_bar=float(d["P_bar"]), T_C=float(d["T_C"]),
            massa_seca_kg=massa_seca_kg, t_s=t * S_POR_MIN, e_exp=e_exp,
            N_kg=N_kg, eps=eps, Q_kg_s=Q_kg_s,
            rho_co2=rho_co2, rho_solido=ent.RHO_PARTICULA, x0=x0,
        )
    if not curvas:
        sys.exit("ERRO: nenhuma curva em CURVAS.")
    return curvas


# =============================================================================
# Ajustes
# =============================================================================


def prever_sovova(c: Curva, Kfa0, Ksa0, sa, yr, t_s=None) -> np.ndarray:
    g = grupos_sovova(
        Kfa0_1_s=Kfa0, Ksa0_1_s=Ksa0, sa=sa, yr_kg_kg=yr, x0_kg_kg=c.x0,
        N_kg=c.N_kg, eps=c.eps, rho_co2_kg_m3=c.rho_co2,
        rho_solido_kg_m3=c.rho_solido, Q_kg_s=c.Q_kg_s,
    )
    q = c.q if t_s is None else c.q_de(t_s)
    return sovova_e(q, x0_kg_kg=c.x0, sa=sa, yr_kg_kg=yr, Z=g.Z, W=g.W,
                    qm=g.qm, qn=g.qn)


class ResiduoSovova:
    """Resíduo relativo de Sovová. Classe (e não closure) para permitir uso
    futuro em multiprocessing, que exige objetos serializáveis."""

    def __init__(self, curva: Curva, yr_imposto: float | None):
        self.c, self.yr = curva, yr_imposto

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        yr = theta[3] if self.yr is None else self.yr
        e = prever_sovova(self.c, theta[0], theta[1], theta[2], yr)
        return (self.c.e_exp - e) / self.c.e_exp


class ResiduoEsquivel:
    def __init__(self, curva: Curva):
        self.c = curva

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        e = esquivel_e(self.c.t_s, e_lim_kg_kg=theta[0], b_s=theta[1])
        return (self.c.e_exp - e) / self.c.e_exp


def params_sovova(yr_livre: bool, prefixo: str = "") -> list[Param]:
    p = f"{prefixo}." if prefixo else ""
    ps = [
        Param(f"{p}Kfa0", KFA0_MIN, KFA0_MAX, "log10", "1/s"),
        Param(f"{p}Ksa0", KSA0_MIN, KSA0_MAX, "log10", "1/s"),
        Param(f"{p}sa", SA_MIN, SA_MAX, "logit10", "-"),
    ]
    if yr_livre:
        ps.append(Param(f"{p}yr", YR_MIN, YR_MAX, "log10", "kg/kg"))
    return ps


def yr_imposto_de(c: Curva) -> float | None:
    if ent.MODO_YR == "legacy":
        return c.yr_primeiro_ponto
    if ent.MODO_YR == "fixed":
        return ent.YR_FIXO_G_KG / G_POR_KG
    return None


def ajustar_sovova(c: Curva, n_starts: int) -> tuple[Ajuste, float]:
    yr_imp = yr_imposto_de(c)
    ps = params_sovova(yr_imp is None)
    res = ResiduoSovova(c, yr_imp)

    def prever(th):
        return prever_sovova(c, th[0], th[1], th[2],
                             th[3] if yr_imp is None else yr_imp)

    aj = ajustar(res, ps, n_starts=n_starts, y_exp=c.e_exp, prever=prever)
    yr = aj.valor("yr") if yr_imp is None else yr_imp
    return aj, yr


def ajustar_esquivel(c: Curva, n_starts: int) -> Ajuste:
    ps = [
        Param("e_lim", E_LIM_MIN, c.x0, "log10", "kg/kg"),  # teto = x0
        Param("b", B_MIN, B_MAX, "log10", "s"),
    ]
    res = ResiduoEsquivel(c)
    return ajustar(res, ps, n_starts=n_starts, y_exp=c.e_exp,
                   prever=lambda th: esquivel_e(c.t_s, e_lim_kg_kg=th[0], b_s=th[1]))


def ajustar_global_sa(curvas: dict[str, Curva], n_starts: int):
    """Ajuste conjunto com um único ``sa`` para todas as curvas."""
    nomes = list(curvas)
    yr_imp = {n: yr_imposto_de(curvas[n]) for n in nomes}
    yr_livre = yr_imp[nomes[0]] is None

    ps = [Param("sa", SA_MIN, SA_MAX, "logit10", "-")]
    for n in nomes:
        ps += [Param(f"{n}.Kfa0", KFA0_MIN, KFA0_MAX, "log10", "1/s"),
               Param(f"{n}.Ksa0", KSA0_MIN, KSA0_MAX, "log10", "1/s")]
        if yr_livre:
            ps.append(Param(f"{n}.yr", YR_MIN, YR_MAX, "log10", "kg/kg"))
    idx = {p.nome: i for i, p in enumerate(ps)}

    def prever_um(n, th):
        yr = th[idx[f"{n}.yr"]] if yr_livre else yr_imp[n]
        return prever_sovova(curvas[n], th[idx[f"{n}.Kfa0"]],
                             th[idx[f"{n}.Ksa0"]], th[0], yr)

    def residuo(th):
        return np.concatenate([(curvas[n].e_exp - prever_um(n, th)) / curvas[n].e_exp
                               for n in nomes])

    aj = ajustar(residuo, ps, n_starts=n_starts)
    por_curva = {}
    for n in nomes:
        e = prever_um(n, aj.theta)
        por_curva[n] = {
            "Kfa0": aj.theta[idx[f"{n}.Kfa0"]],
            "Ksa0": aj.theta[idx[f"{n}.Ksa0"]],
            "yr": aj.theta[idx[f"{n}.yr"]] if yr_livre else yr_imp[n],
            "drm": drm_de((curvas[n].e_exp - e) / curvas[n].e_exp),
            "r2": r2_de(curvas[n].e_exp, e),
        }
    return aj, float(aj.theta[0]), por_curva


# =============================================================================
# Saídas
# =============================================================================


def grafico(c: Curva, aj_s, yr, aj_e, pasta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0.0, c.t_s.max() * 1.05, 400)
    e_s = prever_sovova(c, aj_s.valor("Kfa0"), aj_s.valor("Ksa0"),
                        aj_s.valor("sa"), yr, t_s=t)
    e_e = esquivel_e(t, e_lim_kg_kg=aj_e.valor("e_lim"), b_s=aj_e.valor("b"))
    g = grupos_sovova(
        Kfa0_1_s=aj_s.valor("Kfa0"), Ksa0_1_s=aj_s.valor("Ksa0"),
        sa=aj_s.valor("sa"), yr_kg_kg=yr, x0_kg_kg=c.x0, N_kg=c.N_kg,
        eps=c.eps, rho_co2_kg_m3=c.rho_co2, rho_solido_kg_m3=c.rho_solido,
        Q_kg_s=c.Q_kg_s,
    )

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, x_exp, x_mod, xlabel in (
        (a1, c.t_min, t / S_POR_MIN, "tempo  [min]"),
        (a2, c.q, c.q_de(t), "$q$  [kg CO$_2$ / kg inerte]"),
    ):
        ax.plot(x_exp, c.R_pct, "o", color="#333333", ms=7, zorder=3,
                label="experimental")
        ax.plot(x_mod, 100 * e_s, "-", color="#1f77b4", lw=2,
                label=f"Sovova (DRM {aj_s.drm:.2f} %)")
        ax.plot(x_mod, 100 * e_e, "--", color="#d62728", lw=2,
                label=f"Esquivel (DRM {aj_e.drm:.2f} %)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("rendimento  [%]")
        ax.grid(alpha=0.3)
    for v, estilo, rot in ((g.qm, ":", "$q_m$"), (g.qn, "-.", "$q_n$")):
        if np.isfinite(v) and v <= c.q_de(t).max():
            a2.axvline(v, color="#888888", ls=estilo, lw=1.2)
            a2.annotate(rot, (v, a2.get_ylim()[1]), fontsize=9, ha="center",
                        va="top", color="#888888")
    a1.legend(fontsize=9)
    fig.suptitle(c.rotulo, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(pasta / f"ajuste_{c.nome}.png", dpi=150)
    plt.close(fig)


def csv_curva(c: Curva, aj_s, yr, aj_e, pasta: Path) -> None:
    e_s = prever_sovova(c, aj_s.valor("Kfa0"), aj_s.valor("Ksa0"),
                        aj_s.valor("sa"), yr)
    e_e = esquivel_e(c.t_s, e_lim_kg_kg=aj_e.valor("e_lim"), b_s=aj_e.valor("b"))
    with (pasta / f"ajuste_{c.nome}.csv").open("w", encoding="utf-8") as f:
        f.write("t_min,q,R_exp_pct,R_sovova_pct,R_esquivel_pct,"
                "erro_sovova_pct,erro_esquivel_pct\n")
        for i in range(c.t_s.size):
            obs, s, e_ = c.R_pct[i], 100 * e_s[i], 100 * e_e[i]
            f.write(f"{c.t_min[i]:.6g},{c.q[i]:.6g},{obs:.6g},{s:.6g},{e_:.6g},"
                    f"{100 * (obs - s) / obs:.4g},{100 * (obs - e_) / obs:.4g}\n")


# =============================================================================
# Programa
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perfis", action="store_true",
                    help="calcula o perfil de verossimilhanca (mais lento)")
    ap.add_argument("--sem-graficos", action="store_true")
    ap.add_argument("--sem-solubilidade", action="store_true",
                    help="pula os modelos de Chrastil e del Valle & Aguilera")
    args = ap.parse_args()

    pasta = pasta_saida()
    pasta.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"{ent.MATERIA_PRIMA}")
    print("=" * 72)
    print(f"  teor de oleo (Soxhlet)  {100 * ent.TEOR_OLEO:.3f} %")
    print(f"  x0 (base livre soluto)  {100 * ent.TEOR_OLEO / (1 - ent.TEOR_OLEO):.3f} %"
          f"   <- teto fisico de e_lim")
    print(f"  densidade da particula  {ent.RHO_PARTICULA:.1f} kg/m3")
    print(f"  extrator                {ent.VOLUME_EXTRATOR_ML:.2f} mL "
          f"(inertes {ent.VOLUME_INERTES_ML:.4f} mL)")
    print(f"  densidade do CO2        "
          f"{descrever_fonte(getattr(ent, 'FONTE_DENSIDADE', 'auto'), ent.DENSIDADE_CO2)}")
    print(f"  modo y_r                {ent.MODO_YR}")
    print(f"  curvas                  {len(ent.CURVAS)}")

    curvas = preparar()
    linhas_tab = []

    for nome, c in curvas.items():
        print(f"\n{'-' * 72}\n{c.rotulo}\n{'-' * 72}")
        print(f"  N={c.N_kg * G_POR_KG:.4f} g   eps={c.eps:.4f}   "
              f"Q={c.Q_kg_s * G_POR_KG * S_POR_MIN:.4f} g/min   "
              f"rho={c.rho_co2:.1f} kg/m3")
        print(f"  {c.t_s.size} pontos, recuperou "
              f"{100 * c.recuperado:.2f} % do extraivel")

        aj_s, yr = ajustar_sovova(c, ent.N_PARTIDAS)
        g = grupos_sovova(
            Kfa0_1_s=aj_s.valor("Kfa0"), Ksa0_1_s=aj_s.valor("Ksa0"),
            sa=aj_s.valor("sa"), yr_kg_kg=yr, x0_kg_kg=c.x0, N_kg=c.N_kg,
            eps=c.eps, rho_co2_kg_m3=c.rho_co2, rho_solido_kg_m3=c.rho_solido,
            Q_kg_s=c.Q_kg_s,
        )
        print(f"\n  SOVOVA    Kfa0={aj_s.valor('Kfa0'):.5g} 1/s   "
              f"Ksa0={aj_s.valor('Ksa0'):.5g} 1/s")
        print(f"            sa={aj_s.valor('sa'):.5f}   "
              f"y_r={yr * G_POR_KG:.5f} g/kg")
        print(f"            DRM={aj_s.drm:.4f} %   R2={aj_s.r2:.4f}   "
              f"({aj_s.n_minimos} minimos distintos)")
        print(f"            Z={g.Z:.4g}  W={g.W:.4g}  "
              f"q_m={g.qm:.4g}  q_n={g.qn:.4g}")
        print(f"            q de {c.q.min():.3g} a {c.q.max():.3g}: "
              f"{descrever_periodos(c.q, g.qm, g.qn)}")
        for p in aj_s.mal_determinados():
            lo, hi = aj_s.dispersao[p]
            print(f"    [!] {p} mal determinado: {lo:.3g} a {hi:.3g} "
                  f"({np.log10(hi / lo):.1f} decadas com DRM equivalente)")
        for lim in aj_s.nos_limites:
            print(f"    [!] {lim}")

        aj_e = ajustar_esquivel(c, ent.N_PARTIDAS)
        print(f"\n  ESQUIVEL  e_lim={100 * aj_e.valor('e_lim'):.5f} %   "
              f"b={aj_e.valor('b') / S_POR_MIN:.2f} min")
        print(f"            DRM={aj_e.drm:.4f} %   R2={aj_e.r2:.4f}")
        if esquivel_degenerou(aj_e.valor("b"), c.t_s.max()):
            print(f"    [!] b > t_max ({c.t_min.max():.0f} min): degenerou em "
                  f"reta; so a razao e_lim/b e determinavel")
        for lim in aj_e.nos_limites:
            print(f"    [!] {lim}")

        if args.perfis:
            print("\n  PERFIL DE VEROSSIMILHANCA (Sovova)")
            res = ResiduoSovova(c, yr_imposto_de(c))
            ps = params_sovova(yr_imposto_de(c) is None)
            perfis = perfilar(res, ps, aj_s.theta, n_grade=21, n_starts=4)
            for pn, pf in perfis.items():
                esc = G_POR_KG if pn == "yr" else 1.0
                print(f"    {pn:5s} {pf.status:18s} "
                      f"{pf.faixa[0] * esc:.4g} .. {pf.faixa[1] * esc:.4g}  "
                      f"({pf.largura_decadas:.2f} decadas)")

        if not args.sem_graficos:
            grafico(c, aj_s, yr, aj_e, pasta)
        csv_curva(c, aj_s, yr, aj_e, pasta)

        linhas_tab.append([
            nome, f"{c.P_bar:.0f}", f"{c.T_C:.0f}",
            f"{aj_s.valor('Kfa0'):.5g}", f"{aj_s.valor('Ksa0'):.5g}",
            f"{aj_s.valor('sa'):.5f}", f"{yr * G_POR_KG:.5f}",
            f"{aj_s.drm:.4f}", f"{aj_s.r2:.4f}",
            f"{100 * aj_e.valor('e_lim'):.5f}",
            f"{aj_e.valor('b') / S_POR_MIN:.3f}",
            f"{aj_e.drm:.4f}", f"{aj_e.r2:.4f}",
        ])

    if ent.AJUSTE_GLOBAL_SA and len(curvas) > 1:
        print(f"\n{'=' * 72}\nAJUSTE GLOBAL — um unico sa para todas as curvas"
              f"\n{'=' * 72}")
        aj_g, sa_g, por = ajustar_global_sa(curvas, max(24, ent.N_PARTIDAS // 2))
        drm_ind = np.mean([float(l[7]) for l in linhas_tab])
        drm_glo = np.mean([v["drm"] for v in por.values()])
        print(f"  sa global = {sa_g:.5f}")
        print(f"  DRM medio = {drm_glo:.4f} %  contra {drm_ind:.4f} % individual "
              f"(custo {drm_glo - drm_ind:+.4f} pp, {len(curvas) - 1} parametros a menos)")
        for n, v in por.items():
            print(f"    {n}: Kfa0={v['Kfa0']:.4g}  Ksa0={v['Ksa0']:.4g}  "
                  f"y_r={v['yr'] * G_POR_KG:.4f} g/kg  DRM={v['drm']:.3f} %")

    cab = ["curva", "P_bar", "T_C", "Kfa0_1_s", "Ksa0_1_s", "sa", "yr_g_kg",
           "DRM_sovova_pct", "R2_sovova", "e_lim_pct", "b_min",
           "DRM_esquivel_pct", "R2_esquivel"]
    with (pasta / "parametros.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(cab) + "\n")
        for l in linhas_tab:
            f.write(",".join(l) + "\n")

    # ---- solubilidade: Chrastil e del Valle & Aguilera -------------------
    # Roda por ultimo e num try: e um bloco independente dos modelos
    # cineticos, entao uma falha aqui (CoolProp ausente, por exemplo) nao pode
    # levar junto os resultados de Sovova e Esquivel ja calculados.
    # A flag de linha de comando tem precedencia sobre a de `entrada.py`:
    # --sem-solubilidade desliga mesmo com CALCULAR_SOLUBILIDADE = True.
    fazer_solub = (
        getattr(ent, "CALCULAR_SOLUBILIDADE", True) and not args.sem_solubilidade
    )
    if fazer_solub:
        print()
        try:
            import solubilidade
            solubilidade.main()
        except SystemExit as exc:
            print(f"\n[!] Solubilidade nao calculada: {exc}")
            print("    Os resultados de Sovova e Esquivel acima seguem validos.")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[!] Solubilidade falhou ({type(exc).__name__}: {exc}).")
            print("    Os resultados de Sovova e Esquivel acima seguem validos.")
            print("    Rode 'python solubilidade.py' para ver o erro completo.")

    print(f"\n{'=' * 72}")
    print(f"Saidas em {pasta.resolve()}")
    print("  parametros.csv        tabela consolidada (Sovova + Esquivel)")
    print("  ajuste_<curva>.csv    experimental x calculado, ponto a ponto")
    if not args.sem_graficos:
        print("  ajuste_<curva>.png    curvas, painel duplo e-vs-t e e-vs-q")
    if fazer_solub:
        print("  solubilidade.csv      S experimental e calculada pelos 2 modelos")
        print("  solubilidade_*.png    isobaras e paridade")
    print("\nLembrete: valor pontual so vale para parametro identificavel.")
    print("Onde aparecer [!], reporte faixa ou limite, nao um numero.")


if __name__ == "__main__":
    main()
