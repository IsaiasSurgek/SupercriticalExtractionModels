"""Solubilidade do extrato em CO2 supercritico: Chrastil e del Valle & Aguilera.

Rode com::

    python solubilidade.py

Premissa
--------
Com vazao de CO2 muito baixa o solvente sai saturado do leito, e a solubilidade
pode ser lida direto da curva cinetica:

    S_gg = massa acumulada de extrato / massa acumulada de CO2

**Esta premissa e testavel, e o modulo a testa.** Se o CO2 sai saturado durante
toda a corrida, ``m_extrato`` contra ``m_CO2`` e uma reta pela origem e o
``S_gg`` nao varia ao longo da curva. Onde variar, o ultimo ponto subestima a
solubilidade, e o modulo avisa em vez de reportar o numero em silencio.

Modelos (T em K, S e rho em g/L)
--------------------------------
Chrastil (1982)::              ln S = k*ln(rho) + a/T + b
del Valle & Aguilera (1988)::  ln S = k*ln(rho) + a/T + b + c/T^2

Ambos sao lineares nos parametros depois da linearizacao, entao o ajuste e por
regressao multilinear em ln S — sem otimizador, sem chute inicial, solucao
unica.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import entrada as ent
import main as pipeline
from co2 import coolprop_disponivel, densidade_co2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S_POR_MIN = 60.0
G_POR_KG = 1000.0
LINHA = "=" * 78

# Abaixo disto a reta m_oleo vs m_CO2 deixa de descrever a curva, e o ultimo
# ponto nao e solubilidade de saturacao.
R2_SATURACAO_MIN = 0.95
# Queda aceitavel de S_gg entre o primeiro e o ultimo ponto.
RAZAO_SATURACAO_MIN = 0.85


# =============================================================================
# 1. Solubilidade experimental
# =============================================================================


@dataclass
class SolubExp:
    """Solubilidade experimental de uma condicao.

    Attributes
    ----------
    S_gg : float
        Massa de extrato por massa de CO2 [g/g], do ultimo ponto.
    S_gL : float
        ``S_gg * rho`` [g/L]. Como 1 kg/m3 = 1 g/L, a densidade entra direto.
    r2_reta : float
        R2 do ajuste de ``m_extrato`` contra ``m_CO2`` por reta pela origem.
        Mede se o CO2 saiu saturado durante toda a corrida.
    razao_S : float
        ``S_gg`` do ultimo ponto dividido pelo do primeiro. Perto de 1 confirma
        saturacao; bem abaixo indica que o leito empobreceu e o ultimo ponto
        subestima a solubilidade.
    """

    nome: str
    P_bar: float
    T_C: float
    T_K: float
    rho: float
    S_gg: float
    S_gL: float
    r2_reta: float
    razao_S: float
    m_extrato_g: float
    m_co2_g: float

    @property
    def saturacao_ok(self) -> bool:
        return self.r2_reta >= R2_SATURACAO_MIN and self.razao_S >= RAZAO_SATURACAO_MIN

    @property
    def diagnostico(self) -> str:
        if self.saturacao_ok:
            return "ok"
        partes = []
        if self.r2_reta < R2_SATURACAO_MIN:
            partes.append(f"R2 da reta {self.r2_reta:.3f}")
        if self.razao_S < RAZAO_SATURACAO_MIN:
            partes.append(f"S_gg caiu {100 * (1 - self.razao_S):.0f}%")
        return "SUBESTIMA (" + "; ".join(partes) + ")"


def calcular_experimental() -> list[SolubExp]:
    """Le as curvas de `entrada.py` e calcula a solubilidade de cada condicao."""
    curvas = pipeline.preparar()
    modo = getattr(ent, "FONTE_DENSIDADE", "auto")
    saida: list[SolubExp] = []

    for nome, c in curvas.items():
        m_oleo = c.e_exp * c.N_kg * G_POR_KG          # g, acumulado
        m_co2 = c.Q_kg_s * G_POR_KG * c.t_s           # g, acumulado

        # reta pela origem: m_oleo = a * m_co2
        a = float(np.sum(m_co2 * m_oleo) / np.sum(m_co2**2))
        ss_res = float(np.sum((m_oleo - a * m_co2) ** 2))
        ss_tot = float(np.sum((m_oleo - m_oleo.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        S_ini = float(m_oleo[0] / m_co2[0])
        S_fim = float(m_oleo[-1] / m_co2[-1])
        rho, _ = densidade_co2(c.P_bar, c.T_C, ent.DENSIDADE_CO2, modo)

        saida.append(
            SolubExp(
                nome=nome, P_bar=c.P_bar, T_C=c.T_C, T_K=c.T_C + 273.15,
                rho=rho, S_gg=S_fim, S_gL=S_fim * rho, r2_reta=r2,
                razao_S=S_fim / S_ini,
                m_extrato_g=float(m_oleo[-1]), m_co2_g=float(m_co2[-1]),
            )
        )
    return saida


# =============================================================================
# 2. Ajuste dos modelos
# =============================================================================


@dataclass
class AjusteSolub:
    """Resultado de um ajuste de modelo de solubilidade.

    Attributes
    ----------
    theta : np.ndarray
        Parametros, na ordem de :attr:`nomes`.
    erros : np.ndarray
        Desvios-padrao dos parametros. Com poucos graus de liberdade estes
        numeros sao grandes e pouco confiaveis — e isso e informacao, nao
        defeito.
    dof : int
        Graus de liberdade = n - p.
    cond : float
        Numero de condicao da matriz de delineamento. Acima de ~1e3 indica
        colinearidade entre as colunas, isto e, parametros que os dados nao
        separam.
    """

    modelo: str
    nomes: list[str]
    theta: np.ndarray
    erros: np.ndarray
    n: int
    p: int
    dof: int
    r2_ln: float
    r2_aj_ln: float
    rmse_gL: float
    aard: float
    cond: float
    S_calc: np.ndarray
    avisos: list[str] = field(default_factory=list)

    def valor(self, nome: str) -> float:
        return float(self.theta[self.nomes.index(nome)])


def _matriz(modelo: str, rho: np.ndarray, T_K: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Matriz de delineamento e nomes dos parametros."""
    ln_rho = np.log(rho)
    um = np.ones_like(rho)
    if modelo == "chrastil":
        return np.column_stack([ln_rho, 1.0 / T_K, um]), ["k", "a", "b"]
    if modelo == "dva":
        return (
            np.column_stack([ln_rho, 1.0 / T_K, um, 1.0 / T_K**2]),
            ["k", "a", "b", "c"],
        )
    raise ValueError(f"modelo desconhecido: {modelo!r}")


def ajustar_modelo(modelo: str, exp: list[SolubExp]) -> AjusteSolub:
    """Regressao multilinear em ln S.

    Os dois modelos sao lineares nos parametros apos linearizacao, entao a
    solucao e fechada: nao ha chute inicial nem minimo local. O que pode dar
    errado e outra coisa — graus de liberdade insuficientes e colinearidade
    entre colunas —, e e isso que os avisos cobrem.
    """
    rho = np.array([e.rho for e in exp])
    T_K = np.array([e.T_K for e in exp])
    S = np.array([e.S_gL for e in exp])
    y = np.log(S)

    X, nomes = _matriz(modelo, rho, T_K)
    n, p = X.shape
    dof = n - p

    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ theta

    # erros-padrao dos parametros
    if dof > 0:
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        erros = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    else:
        erros = np.full(p, np.nan)

    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_aj = (
        1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    )

    S_calc = np.exp(X @ theta)
    rmse = float(np.sqrt(np.mean((S - S_calc) ** 2)))
    aard = float(100.0 * np.mean(np.abs(S - S_calc) / S))
    cond = float(np.linalg.cond(X))

    avisos: list[str] = []
    if n <= p:
        avisos.append(
            f"n = {n} condicoes para p = {p} parametros: o ajuste NAO tem "
            f"graus de liberdade e e apenas interpolativo. Nao extrapole."
        )
    elif dof < 3:
        avisos.append(
            f"apenas {dof} grau(s) de liberdade (n={n}, p={p}): R2 e erros-padrao "
            f"sao pouco informativos e o ajuste tende a interpolar o ruido."
        )
    if cond > 1e3:
        avisos.append(
            f"numero de condicao {cond:.3g}: as colunas do delineamento sao "
            f"quase colineares, entao os parametros nao sao separaveis."
        )
    if modelo == "dva":
        r = float(np.corrcoef(1.0 / T_K, 1.0 / T_K**2)[0, 1])
        if abs(r) > 0.999:
            avisos.append(
                f"1/T e 1/T^2 tem correlacao {r:.6f} nesta faixa de temperatura: "
                f"o termo c/T^2 e praticamente redundante com a/T, e o parametro "
                f"'c' nao e determinado pelos dados."
            )

    nome_bonito = (
        "Chrastil (1982)" if modelo == "chrastil" else "del Valle & Aguilera (1988)"
    )
    return AjusteSolub(
        modelo=nome_bonito, nomes=nomes, theta=theta, erros=erros, n=n, p=p,
        dof=dof, r2_ln=r2, r2_aj_ln=r2_aj, rmse_gL=rmse, aard=aard, cond=cond,
        S_calc=S_calc, avisos=avisos,
    )


def prever(aj: AjusteSolub, rho: np.ndarray, T_K: np.ndarray) -> np.ndarray:
    """S [g/L] previsto pelo modelo ajustado."""
    modelo = "chrastil" if aj.p == 3 else "dva"
    X, _ = _matriz(modelo, np.atleast_1d(rho), np.atleast_1d(T_K))
    return np.exp(X @ aj.theta)


# =============================================================================
# 3. Saidas
# =============================================================================


def grafico_isobaras(
    exp: list[SolubExp], ajs: list[AjusteSolub], destino: Path
) -> None:
    """S (g/L) contra T (°C), pontos experimentais e curvas dos modelos.

    Uma isobara por pressao presente nos dados. Isobaras com um unico ponto
    experimental sao tracejadas: a curva ali e extrapolacao do modelo global,
    nao ajuste local, e nada nos dados daquela pressao a sustenta.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tem_coolprop = coolprop_disponivel()
    if not tem_coolprop:
        print("  [!] CoolProp ausente: as curvas continuas precisam de rho(T,P)")
        print("      numa malha fina de temperatura. Gerando so os pontos.")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    cores = plt.cm.viridis(np.linspace(0.05, 0.85, len({e.P_bar for e in exp})))
    pressoes = sorted({e.P_bar for e in exp})
    T_mesh = np.linspace(
        min(e.T_C for e in exp) - 3, max(e.T_C for e in exp) + 3, 200
    )

    for cor, P in zip(cores, pressoes):
        pontos = [e for e in exp if e.P_bar == P]
        ax.plot([e.T_C for e in pontos], [e.S_gL for e in pontos], "o",
                color=cor, ms=9, mec="black", mew=0.8, zorder=5,
                label=f"exp. {P:.0f} bar")

        if not tem_coolprop:
            continue
        from CoolProp.CoolProp import PropsSI

        rho_mesh = np.array([
            PropsSI("D", "P", P * 1e5, "T", T + 273.15, "CO2") for T in T_mesh
        ])
        T_K_mesh = T_mesh + 273.15
        so_um = len(pontos) == 1
        for aj, estilo in zip(ajs, ("-", "--")):
            ax.plot(T_mesh, prever(aj, rho_mesh, T_K_mesh), estilo, color=cor,
                    lw=1.9 if not so_um else 1.1,
                    alpha=1.0 if not so_um else 0.55)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", ls="-", lw=1.9),
                Line2D([], [], color="0.3", ls="--", lw=1.9)]
    labels += ["Chrastil", "del Valle & Aguilera"]
    ax.legend(handles, labels, fontsize=8.5, ncol=2, framealpha=0.95)

    ax.set_xlabel("Temperatura  [°C]", fontsize=11)
    ax.set_ylabel("Solubilidade  [g extrato / L CO$_2$]", fontsize=11)
    ax.set_title("Solubilidade do extrato em CO$_2$ supercrítico",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, ls=":")
    n_um = sum(1 for P in pressoes if sum(1 for e in exp if e.P_bar == P) == 1)
    if n_um:
        ax.text(0.02, 0.02,
                f"linhas tênues: {n_um} isóbaras com um único ponto experimental "
                f"(extrapolação do modelo global)",
                transform=ax.transAxes, fontsize=7.5, style="italic", color="0.35")
    fig.tight_layout()
    fig.savefig(destino, dpi=300)
    plt.close(fig)


def grafico_paridade(exp: list[SolubExp], ajs: list[AjusteSolub], destino: Path) -> None:
    """S_exp contra S_calc, com y = x e faixa de +-10 %."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S_exp = np.array([e.S_gL for e in exp])
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    lo, hi = 0.75 * S_exp.min(), 1.3 * S_exp.max()
    reta = np.array([lo, hi])
    ax.plot(reta, reta, "-", color="black", lw=1.2, label="y = x")
    ax.fill_between(reta, 0.9 * reta, 1.1 * reta, color="0.6", alpha=0.18,
                    label="±10 %")
    for aj, marcador, cor in zip(ajs, ("o", "s"), ("#1f77b4", "#d62728")):
        ax.plot(S_exp, aj.S_calc, marcador, color=cor, ms=8, mec="black",
                mew=0.7, alpha=0.9,
                label=f"{aj.modelo.split(' (')[0]}  (AARD {aj.aard:.1f} %)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("$S_{exp}$  [g/L]", fontsize=11)
    ax.set_ylabel("$S_{calc}$  [g/L]", fontsize=11)
    ax.set_title("Paridade", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(destino, dpi=300)
    plt.close(fig)


def escrever_csv(exp: list[SolubExp], ajs: list[AjusteSolub], destino: Path) -> None:
    ch, dva = ajs
    with destino.open("w", encoding="utf-8") as f:
        f.write("condicao,T_C,P_bar,rho_CO2_g_L,S_exp_g_g,S_exp_g_L,"
                "S_chrastil_g_L,S_dVA_g_L,desvio_chrastil_pct,desvio_dVA_pct,"
                "saturacao_r2,saturacao_razao_S,diagnostico\n")
        for i, e in enumerate(exp):
            dc = 100 * (ch.S_calc[i] - e.S_gL) / e.S_gL
            dd = 100 * (dva.S_calc[i] - e.S_gL) / e.S_gL
            f.write(f"{e.nome},{e.T_C:.6g},{e.P_bar:.6g},{e.rho:.6g},"
                    f"{e.S_gg:.6g},{e.S_gL:.6g},{ch.S_calc[i]:.6g},"
                    f"{dva.S_calc[i]:.6g},{dc:.4g},{dd:.4g},"
                    f"{e.r2_reta:.4g},{e.razao_S:.4g},{e.diagnostico}\n")


# =============================================================================
# Programa
# =============================================================================


def main() -> None:
    pasta = pipeline.pasta_saida()
    pasta.mkdir(parents=True, exist_ok=True)

    exp = calcular_experimental()
    n = len(exp)

    print(LINHA)
    print("SOLUBILIDADE EM CO2 SUPERCRITICO")
    print(LINHA)
    print(f"  {ent.MATERIA_PRIMA}")
    print(f"  {n} condicoes experimentais\n")

    # ---- 1. experimental ------------------------------------------------
    print("1. SOLUBILIDADE EXPERIMENTAL (ultimo ponto de cada curva)")
    print(f"  {'cond':6s} {'T[C]':>6s} {'P[bar]':>7s} {'rho[g/L]':>9s} "
          f"{'m_ext[g]':>9s} {'m_CO2[g]':>9s} {'S[g/g]':>9s} {'S[g/L]':>8s}")
    for e in exp:
        print(f"  {e.nome:6s} {e.T_C:6.0f} {e.P_bar:7.0f} {e.rho:9.2f} "
              f"{e.m_extrato_g:9.4f} {e.m_co2_g:9.4f} {e.S_gg:9.5f} {e.S_gL:8.3f}")

    print("\n2. A PREMISSA DE SATURACAO SE SUSTENTA?")
    print("   Se o CO2 sai saturado o tempo todo, m_extrato x m_CO2 e reta pela")
    print("   origem e S_gg nao varia ao longo da curva.\n")
    print(f"  {'cond':6s} {'R2 reta':>9s} {'S_fim/S_ini':>12s}   diagnostico")
    ruins = []
    for e in exp:
        if not e.saturacao_ok:
            ruins.append(e)
        print(f"  {e.nome:6s} {e.r2_reta:9.4f} {e.razao_S:12.3f}   {e.diagnostico}")
    if ruins:
        print(f"\n  [!] {len(ruins)} condicao(oes) NAO satisfazem a premissa: "
              f"{', '.join(e.nome for e in ruins)}.")
        print("      Nelas o leito empobreceu antes do fim da corrida, entao o")
        print("      ultimo ponto e MENOR que a solubilidade de saturacao. O")
        print("      valor entra no ajuste mesmo assim, como voce pediu, mas")
        print("      puxa a curva para baixo e infla o AARD.")

    # ---- 2. ajustes ------------------------------------------------------
    ajs = [ajustar_modelo("chrastil", exp), ajustar_modelo("dva", exp)]

    print(f"\n3. PARAMETROS AJUSTADOS")
    for aj in ajs:
        print(f"\n  {aj.modelo}   (n={aj.n}, p={aj.p}, dof={aj.dof})")
        for nome, v, s in zip(aj.nomes, aj.theta, aj.erros):
            rel = f"  ({100 * abs(s / v):.0f} %)" if np.isfinite(s) and v != 0 else ""
            s_txt = f"{s:.4g}" if np.isfinite(s) else "n/d"
            print(f"    {nome:2s} = {v:12.5g}  +- {s_txt:>10s}{rel}")
        for a in aj.avisos:
            print(f"    [!] {a}")

    print(f"\n4. QUALIDADE DO AJUSTE")
    print(f"  {'modelo':30s} {'R2(lnS)':>9s} {'R2aj':>8s} {'RMSE[g/L]':>10s} "
          f"{'AARD%':>7s} {'DRM%':>7s}")
    for aj in ajs:
        S_exp = np.array([e.S_gL for e in exp])
        drm = 100 * np.mean(np.abs(S_exp - aj.S_calc) / S_exp)
        r2aj = f"{aj.r2_aj_ln:8.4f}" if np.isfinite(aj.r2_aj_ln) else "     n/d"
        print(f"  {aj.modelo:30s} {aj.r2_ln:9.4f} {r2aj} {aj.rmse_gL:10.4f} "
              f"{aj.aard:7.2f} {drm:7.2f}")

    print("\n  Desvio relativo por condicao (ultimo ponto de cada curva):")
    print(f"  {'cond':6s} {'S_exp[g/L]':>11s} {'Chrastil':>10s} {'dev%':>8s} "
          f"{'dVA':>10s} {'dev%':>8s}")
    for i, e in enumerate(exp):
        dc = 100 * (ajs[0].S_calc[i] - e.S_gL) / e.S_gL
        dd = 100 * (ajs[1].S_calc[i] - e.S_gL) / e.S_gL
        print(f"  {e.nome:6s} {e.S_gL:11.4f} {ajs[0].S_calc[i]:10.4f} {dc:+8.2f} "
              f"{ajs[1].S_calc[i]:10.4f} {dd:+8.2f}")

    # ---- 3. saidas -------------------------------------------------------
    p_iso = pasta / "solubilidade_vs_temperatura.png"
    p_par = pasta / "solubilidade_paridade.png"
    p_csv = pasta / "solubilidade.csv"
    grafico_isobaras(exp, ajs, p_iso)
    grafico_paridade(exp, ajs, p_par)
    escrever_csv(exp, ajs, p_csv)

    print(f"\n5. ARQUIVOS")
    for p in (p_iso, p_par, p_csv):
        print(f"  {p}")

    # ---- 4. leitura ------------------------------------------------------
    print(f"\n{LINHA}\nCOMO LER ESTES NUMEROS\n{LINHA}")
    melhor = min(ajs, key=lambda a: a.aard)
    outro = ajs[1] if melhor is ajs[0] else ajs[0]
    print(f"  O {melhor.modelo.split(' (')[0]} tem AARD menor "
          f"({melhor.aard:.2f} % contra {outro.aard:.2f} %).")
    if ajs[1].aard < ajs[0].aard and ajs[1].dof < 3:
        print("  Mas o del Valle & Aguilera tem um parametro a mais e "
              f"{ajs[1].dof} grau(s) de liberdade: ajustar melhor com mais")
        print("  parametros e o esperado, nao evidencia de ser o modelo certo.")
    print(f"\n  Pressoes com uma unica temperatura: "
          f"{sorted(P for P in {e.P_bar for e in exp} if sum(1 for e in exp if e.P_bar == P) == 1)}")
    print("  Nelas a isobara do grafico e extrapolacao do modelo global; nao ha")
    print("  dado naquela pressao que a sustente.")


if __name__ == "__main__":
    main()
