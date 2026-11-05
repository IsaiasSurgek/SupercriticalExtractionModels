"""Relatório consolidado em PDF.

Rode com::

    python relatorio_pdf.py

Gera ``relatorio.pdf`` na pasta de saída, com tudo que a ferramenta calcula:
configuração adotada, parâmetros de Sovová e Esquível por condição, diagnóstico
de identificabilidade, ajuste global com ``sa`` compartilhado, solubilidade
pelos modelos de Chrastil e del Valle & Aguilera, e todos os gráficos.

Usa apenas ``matplotlib.backends.backend_pdf``, então não acrescenta nenhuma
dependência além das que a ferramenta já exige.

É autossuficiente: refaz os ajustes por conta própria, e não depende de o
``main.py`` ter rodado antes. Rodar os dois refaz as contas duas vezes — se
quiser só o PDF, rode apenas este arquivo.
"""

from __future__ import annotations

import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import entrada as ent  # noqa: E402
import main as pipeline  # noqa: E402
import solubilidade as sol  # noqa: E402
from co2 import coolprop_disponivel, descrever_fonte  # noqa: E402
from modelos import (  # noqa: E402
    descrever_periodos, esquivel_degenerou, esquivel_e, grupos_sovova,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S_POR_MIN = 60.0
G_POR_KG = 1000.0
A4 = (8.27, 11.69)          # retrato, polegadas
A4_L = (11.69, 8.27)        # paisagem

CORES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f"]


# =============================================================================
# Paginas de texto
# =============================================================================


def _pagina(pdf: PdfPages, titulo: str, subtitulo: str = "") -> tuple:
    """Abre uma pagina A4 retrato com cabecalho, devolve (fig, y_atual)."""
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.955, titulo, fontsize=15, fontweight="bold", va="top")
    y = 0.925
    if subtitulo:
        fig.text(0.08, y, subtitulo, fontsize=9, color="0.35", va="top")
        y -= 0.022
    fig.add_artist(plt.Line2D([0.08, 0.92], [y, y], color="0.75", lw=0.8))
    return fig, y - 0.028


def _texto(fig, y: float, txt: str, *, tam: float = 8.2, mono: bool = False,
           cor: str = "black", negrito: bool = False, x: float = 0.08) -> float:
    """Escreve um bloco e devolve o novo y."""
    n = txt.count("\n") + 1
    fig.text(x, y, txt, fontsize=tam, va="top", color=cor,
             family="monospace" if mono else "sans-serif",
             fontweight="bold" if negrito else "normal")
    return y - n * (tam / 640.0) - 0.012


def _tabela(fig, y: float, cab: list[str], linhas: list[list[str]],
            larguras: list[int], *, tam: float = 7.6) -> float:
    """Tabela monoespacada com colunas de largura fixa."""
    def fmt(campos):
        return "  ".join(str(c).ljust(w)[:w] if i == 0 else str(c).rjust(w)[:w]
                         for i, (c, w) in enumerate(zip(campos, larguras)))

    y = _texto(fig, y, fmt(cab), tam=tam, mono=True, negrito=True)
    y += 0.006
    fig.add_artist(plt.Line2D([0.08, 0.92], [y, y], color="0.85", lw=0.6))
    y -= 0.008
    for L in linhas:
        y = _texto(fig, y, fmt(L), tam=tam, mono=True)
        y += 0.004
    return y - 0.012


def _fechar(pdf: PdfPages, fig) -> None:
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Conteudo
# =============================================================================


def pagina_capa(pdf: PdfPages, curvas: dict) -> None:
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.72, "Extração supercrítica com CO$_2$", fontsize=22,
             fontweight="bold", ha="center")
    fig.text(0.5, 0.665, "Modelagem cinética e de solubilidade", fontsize=14,
             ha="center", color="0.3")
    fig.add_artist(plt.Line2D([0.25, 0.75], [0.635, 0.635], color="0.6", lw=1.2))
    fig.text(0.5, 0.585, ent.MATERIA_PRIMA, fontsize=13, ha="center")
    fig.text(0.5, 0.545, f"{len(curvas)} curvas cinéticas", fontsize=10,
             ha="center", color="0.4")

    cond = "\n".join(
        f"{c.nome}:  {c.P_bar:.0f} bar / {c.T_C:.0f} °C   —   "
        f"{c.t_s.size} pontos até {c.t_min.max():.0f} min"
        for c in curvas.values()
    )
    fig.text(0.5, 0.46, cond, fontsize=9.5, ha="center", va="top",
             family="monospace", color="0.25")

    fig.text(0.5, 0.20,
             "Modelos: Sovová (1994) · Esquível et al. (1999)\n"
             "Chrastil (1982) · del Valle & Aguilera (1988)",
             fontsize=9, ha="center", color="0.45")
    fig.text(0.5, 0.10, date.today().strftime("%d/%m/%Y"), fontsize=9,
             ha="center", color="0.55")
    _fechar(pdf, fig)


def pagina_config(pdf: PdfPages, curvas: dict) -> None:
    fig, y = _pagina(pdf, "1. Configuração adotada",
                     "tudo vem de entrada.py; nada é hard-coded no cálculo")
    c0 = next(iter(curvas.values()))
    x0 = ent.TEOR_OLEO / (1 - ent.TEOR_OLEO)
    Q = pipeline.vazao_g_min(c0.massa_seca_kg * 1000)

    y = _texto(fig, y, "Matéria-prima", negrito=True, tam=10)
    y = _tabela(fig, y, ["grandeza", "valor", "observação"],
                [["teor de óleo (Soxhlet)", f"{100*ent.TEOR_OLEO:.3f} %",
                  "define x0, N e o teto de e_lim"],
                 ["x0 (base livre soluto)", f"{100*x0:.3f} %",
                  "teto físico de e_lim"],
                 ["densidade da partícula", f"{ent.RHO_PARTICULA:.1f} kg/m3",
                  "entra só na porosidade"],
                 ["diâmetro médio", f"{ent.DIAMETRO_PARTICULA_MM:.3f} mm",
                  "não entra no ajuste"]],
                [26, 16, 34])

    y = _texto(fig, y, "Equipamento", negrito=True, tam=10)
    porosidades = [c.eps for c in curvas.values()]
    y = _tabela(fig, y, ["grandeza", "valor", "observação"],
                [["volume do extrator", f"{ent.VOLUME_EXTRATOR_ML:.2f} mL", ""],
                 ["inertes no leito", f"{ent.VOLUME_INERTES_ML:.4f} mL",
                  "esferas de vidro"],
                 ["porosidade", f"{min(porosidades):.4f}-{max(porosidades):.4f}",
                  "calculada por curva"],
                 ["vazão de CO2", f"{Q:.5f} g/min",
                  f"bolhômetro {ent.VAZAO_BOLHOMETRO_ML_MIN} mL/min"],
                 ["densidade do CO2", "", descrever_fonte(
                     getattr(ent, "FONTE_DENSIDADE", "auto"), ent.DENSIDADE_CO2)]],
                [26, 16, 34])

    y = _texto(fig, y, "Regime de operação", negrito=True, tam=10)
    massa_leito = c0.eps * ent.VOLUME_EXTRATOR_ML * 1e-6 * c0.rho_co2 * 1000
    horas = massa_leito / (Q / 60.0) / 60.0
    y = _texto(fig, y,
        f"A vazão medida corresponde a {Q/ (c0.massa_seca_kg*1000):.5f} g CO2 por g de matriz\n"
        f"por minuto. Com {massa_leito:.1f} g de CO2 dentro do extrator, trocar o conteúdo\n"
        f"do leito uma vez levaria {horas:.0f} h, contra corridas de "
        f"{min(c.t_min.max() for c in curvas.values())/60:.1f} a "
        f"{max(c.t_min.max() for c in curvas.values())/60:.1f} h.\n\n"
        f"Isto não é extração dinâmica convencional: é extração quase estática com sangria\n"
        f"lenta. O CO2 tem tempo de sobra para saturar, o que explica a solubilidade na faixa\n"
        f"de equilíbrio e, ao mesmo tempo, a recuperação baixa — passou pouco solvente.",
        tam=8.6)

    y = _texto(fig, y, "Recuperação", negrito=True, tam=10)
    _tabela(fig, y, ["curva", "extraído [g]", "% do extraível", "q final"],
            [[c.nome, f"{c.e_exp[-1]*c.N_kg*1000:.4f}",
              f"{100*c.recuperado:.2f}", f"{c.q.max():.3f}"]
             for c in curvas.values()], [10, 14, 16, 10])
    _fechar(pdf, fig)


def pagina_cinetica(pdf: PdfPages, res: list[dict]) -> None:
    fig, y = _pagina(pdf, "2. Parâmetros cinéticos",
                     "Sovová (1994) e Esquível et al. (1999), por condição")

    y = _texto(fig, y, "Sovová (1994)", negrito=True, tam=10)
    y = _tabela(fig, y,
                ["curva", "P/T", "Kfa0 [1/s]", "Ksa0 [1/s]", "sa", "y_r[g/kg]",
                 "DRM%", "R2"],
                [[r["nome"], f"{r['P']:.0f}/{r['T']:.0f}",
                  f"{r['Kfa0']:.4g}", f"{r['Ksa0']:.4g}", f"{r['sa']:.4f}",
                  f"{r['yr']*G_POR_KG:.3f}", f"{r['drm_s']:.3f}",
                  f"{r['r2_s']:.4f}"] for r in res],
                [7, 9, 11, 11, 8, 10, 7, 7])

    y = _texto(fig, y, "Grupos adimensionais e períodos cobertos",
               negrito=True, tam=10)
    y = _tabela(fig, y, ["curva", "Z", "W", "q_m", "q_n", "q observado", "períodos"],
                [[r["nome"], f"{r['Z']:.4g}", f"{r['W']:.4g}", f"{r['qm']:.4g}",
                  f"{r['qn']:.4g}", f"{r['q_min']:.3g}-{r['q_max']:.3g}",
                  r["periodos"]] for r in res],
                [7, 10, 10, 10, 10, 14, 10])

    y = _texto(fig, y, "Esquível et al. (1999)", negrito=True, tam=10)
    y = _tabela(fig, y,
                ["curva", "P/T", "e_lim [%]", "b [min]", "t_máx[min]", "DRM%", "R2"],
                [[r["nome"], f"{r['P']:.0f}/{r['T']:.0f}",
                  f"{100*r['e_lim']:.4f}", f"{r['b']/S_POR_MIN:.2f}",
                  f"{r['t_max']:.0f}", f"{r['drm_e']:.3f}", f"{r['r2_e']:.4f}"]
                 for r in res],
                [7, 9, 12, 12, 12, 8, 8])

    y = _texto(fig, y,
        "e_lim tem teto físico igual a x0: não pode assintotar acima do óleo que existe\n"
        "na matriz. Quando b excede o tempo do ensaio o modelo degenera em reta, e só a\n"
        "razão e_lim/b é determinável — reportar os dois separadamente sugere uma precisão\n"
        "que os dados não têm.", tam=8.2, cor="0.3")
    _fechar(pdf, fig)


def pagina_diagnostico(pdf: PdfPages, res: list[dict], glob) -> None:
    fig, y = _pagina(pdf, "3. Identificabilidade",
                     "o que os dados determinam, e o que não determinam")

    y = _texto(fig, y,
        "Um parâmetro só é determinado se variá-lo piorar o ajuste. Onde a superfície é\n"
        "plana, o valor que sai é escolha do otimizador, não informação dos dados. A coluna\n"
        "abaixo mostra a faixa que soluções de DRM equivalente cobrem.", tam=8.4, cor="0.3")

    linhas = []
    for r in res:
        for p, (lo, hi) in sorted(r["dispersao"].items()):
            dec = np.log10(hi / lo) if lo > 0 else np.inf
            esc = G_POR_KG if p == "yr" else 1.0
            veredito = "determinado" if dec <= 1 else (
                "fraco" if dec <= 2 else "NÃO determinado")
            linhas.append([r["nome"], p, f"{lo*esc:.4g}", f"{hi*esc:.4g}",
                           f"{dec:.1f}", veredito])
    y = _tabela(fig, y,
                ["curva", "parâmetro", "mínimo", "máximo", "décadas", "veredito"],
                linhas, [7, 10, 12, 12, 8, 16], tam=7.0)

    avisos = [a for r in res for a in r["avisos"]]
    if avisos:
        y = _texto(fig, y, "Avisos emitidos", negrito=True, tam=10)
        y = _texto(fig, y, "\n".join(f"• {a}" for a in avisos), tam=7.8,
                   cor="#a01010")

    if glob is not None:
        y = _texto(fig, y, "Ajuste global — um único sa para todas as curvas",
                   negrito=True, tam=10)
        drm_ind = float(np.mean([r["drm_s"] for r in res]))
        y = _texto(fig, y,
            f"sa global = {glob['sa']:.5f}\n"
            f"DRM médio = {glob['drm']:.3f} % contra {drm_ind:.3f} % dos ajustes\n"
            f"individuais — custo de {glob['drm']-drm_ind:+.3f} pp com "
            f"{len(res)-1} parâmetros a menos.\n\n"
            f"sa é propriedade da moagem: mesmo lote e mesma peneira devem dar o mesmo\n"
            f"valor. Um sa variando entre condições costuma ser artefato de ajuste.",
            tam=8.4)
        _tabela(fig, y, ["curva", "Kfa0 [1/s]", "Ksa0 [1/s]", "y_r [g/kg]", "DRM%"],
                [[n, f"{v['Kfa0']:.4g}", f"{v['Ksa0']:.4g}",
                  f"{v['yr']*G_POR_KG:.3f}", f"{v['drm']:.3f}"]
                 for n, v in glob["por_curva"].items()], [10, 13, 13, 12, 8])
    _fechar(pdf, fig)


def pagina_solubilidade(pdf: PdfPages, exp: list, ajs: list) -> None:
    fig, y = _pagina(pdf, "4. Solubilidade",
                     "do último ponto de cada curva; modelos de Chrastil e del Valle")

    y = _texto(fig, y, "Solubilidade experimental", negrito=True, tam=10)
    y = _tabela(fig, y,
                ["cond", "T[°C]", "P[bar]", "rho[g/L]", "m_ext[g]", "m_CO2[g]",
                 "S[g/g]", "S[g/L]"],
                [[e.nome, f"{e.T_C:.0f}", f"{e.P_bar:.0f}", f"{e.rho:.2f}",
                  f"{e.m_extrato_g:.4f}", f"{e.m_co2_g:.4f}", f"{e.S_gg:.5f}",
                  f"{e.S_gL:.3f}"] for e in exp],
                [6, 7, 8, 10, 10, 10, 9, 9])

    y = _texto(fig, y, "A premissa de saturação se sustenta?", negrito=True, tam=10)
    y = _texto(fig, y,
        "Se o CO2 sai saturado o tempo todo, m_extrato contra m_CO2 é reta pela origem e\n"
        "S_gg não varia ao longo da curva. Onde varia, o leito empobreceu e o último ponto\n"
        "SUBESTIMA a solubilidade.", tam=8.2, cor="0.3")
    y = _tabela(fig, y, ["cond", "R2 da reta", "S_fim/S_ini", "diagnóstico"],
                [[e.nome, f"{e.r2_reta:.4f}", f"{e.razao_S:.3f}", e.diagnostico]
                 for e in exp], [8, 12, 13, 34])

    y = _texto(fig, y, "Parâmetros ajustados", negrito=True, tam=10)
    for aj in ajs:
        y = _texto(fig, y, f"{aj.modelo}   (n={aj.n}, p={aj.p}, dof={aj.dof})",
                   tam=8.6, negrito=True)
        txt = "   ".join(
            f"{nm} = {v:.5g} ± {s:.3g}" if np.isfinite(s) else f"{nm} = {v:.5g}"
            for nm, v, s in zip(aj.nomes, aj.theta, aj.erros))
        y = _texto(fig, y, "  " + txt, tam=7.8, mono=True)
        for a in aj.avisos:
            y = _texto(fig, y, f"  [!] {a}", tam=7.2, cor="#a01010")

    y = _texto(fig, y, "Qualidade do ajuste", negrito=True, tam=10)
    S_exp = np.array([e.S_gL for e in exp])
    y = _tabela(fig, y, ["modelo", "R2(lnS)", "R2 aj.", "RMSE[g/L]", "AARD%"],
                [[aj.modelo, f"{aj.r2_ln:.4f}",
                  f"{aj.r2_aj_ln:.4f}" if np.isfinite(aj.r2_aj_ln) else "n/d",
                  f"{aj.rmse_gL:.4f}", f"{aj.aard:.2f}"] for aj in ajs],
                [30, 10, 10, 11, 8])

    y = _tabela(fig, y, ["cond", "S_exp[g/L]", "Chrastil", "dev%", "dVA", "dev%"],
                [[e.nome, f"{e.S_gL:.4f}", f"{ajs[0].S_calc[i]:.4f}",
                  f"{100*(ajs[0].S_calc[i]-e.S_gL)/e.S_gL:+.2f}",
                  f"{ajs[1].S_calc[i]:.4f}",
                  f"{100*(ajs[1].S_calc[i]-e.S_gL)/e.S_gL:+.2f}"]
                 for i, e in enumerate(exp)], [7, 12, 11, 8, 11, 8])
    _fechar(pdf, fig)


# =============================================================================
# Graficos
# =============================================================================


def paginas_curvas(pdf: PdfPages, curvas: dict, res: list[dict]) -> None:
    """Uma pagina paisagem por curva: e-vs-t e e-vs-q lado a lado."""
    for r in res:
        c = curvas[r["nome"]]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=A4_L)
        t = np.linspace(0.0, c.t_s.max() * 1.05, 400)
        e_s = pipeline.prever_sovova(c, r["Kfa0"], r["Ksa0"], r["sa"], r["yr"], t_s=t)
        e_e = esquivel_e(t, e_lim_kg_kg=r["e_lim"], b_s=r["b"])

        for ax, xe, xm, xl in (
            (a1, c.t_min, t / S_POR_MIN, "tempo  [min]"),
            (a2, c.q, c.q_de(t), "$q$  [kg CO$_2$ / kg inerte]"),
        ):
            ax.plot(xe, c.R_pct, "o", color="#222", ms=7, zorder=5,
                    label="experimental")
            ax.plot(xm, 100 * e_s, "-", color="#1f77b4", lw=2,
                    label=f"Sovová (DRM {r['drm_s']:.2f} %)")
            ax.plot(xm, 100 * e_e, "--", color="#d62728", lw=2,
                    label=f"Esquível (DRM {r['drm_e']:.2f} %)")
            ax.set_xlabel(xl)
            ax.set_ylabel("rendimento  [%, base livre de soluto]")
            ax.grid(alpha=0.3, ls=":")
        for v, es, rot in ((r["qm"], ":", "$q_m$"), (r["qn"], "-.", "$q_n$")):
            if np.isfinite(v) and v <= c.q_de(t).max():
                a2.axvline(v, color="#888", ls=es, lw=1.2)
                a2.annotate(rot, (v, a2.get_ylim()[1]), fontsize=9, ha="center",
                            va="top", color="#888")
        a1.legend(fontsize=9)
        fig.suptitle(f"{c.nome} — {c.P_bar:.0f} bar / {c.T_C:.0f} °C",
                     fontsize=13, fontweight="bold")
        fig.text(0.5, 0.02, descrever_periodos(c.q, r["qm"], r["qn"]),
                 ha="center", fontsize=8.5, color="0.35", style="italic")
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        _fechar(pdf, fig)


def pagina_todas_curvas(pdf: PdfPages, curvas: dict) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=A4_L)
    for cor, (n, c) in zip(CORES, curvas.items()):
        rot = f"{n} ({c.P_bar:.0f} bar / {c.T_C:.0f} °C)"
        a1.plot(c.t_min, c.R_pct, "o-", color=cor, ms=5, lw=1.3, label=rot)
        a2.plot(c.q, c.R_pct, "o-", color=cor, ms=5, lw=1.3, label=rot)
    a1.set_xlabel("tempo  [min]")
    a2.set_xlabel("$q$  [kg CO$_2$ / kg inerte]")
    for ax in (a1, a2):
        ax.set_ylabel("rendimento  [%]")
        ax.grid(alpha=0.3, ls=":")
    a1.legend(fontsize=8.5)
    fig.suptitle("Todas as condições: tempo e massa específica de solvente",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02,
             "colapso das curvas no painel da direita indicaria controle por solubilidade",
             ha="center", fontsize=8.5, color="0.35", style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _fechar(pdf, fig)


def paginas_solubilidade(pdf: PdfPages, exp: list, ajs: list) -> None:
    # --- isobaras ---
    fig, ax = plt.subplots(figsize=A4_L)
    pressoes = sorted({e.P_bar for e in exp})
    cores = plt.cm.viridis(np.linspace(0.05, 0.85, len(pressoes)))
    T_mesh = np.linspace(min(e.T_C for e in exp) - 3,
                         max(e.T_C for e in exp) + 3, 200)
    for cor, P in zip(cores, pressoes):
        pts = [e for e in exp if e.P_bar == P]
        ax.plot([e.T_C for e in pts], [e.S_gL for e in pts], "o", color=cor,
                ms=10, mec="black", mew=0.8, zorder=5, label=f"exp. {P:.0f} bar")
        if not coolprop_disponivel():
            continue
        from CoolProp.CoolProp import PropsSI
        rho = np.array([PropsSI("D", "P", P * 1e5, "T", T + 273.15, "CO2")
                        for T in T_mesh])
        so_um = len(pts) == 1
        for aj, es in zip(ajs, ("-", "--")):
            ax.plot(T_mesh, sol.prever(aj, rho, T_mesh + 273.15), es, color=cor,
                    lw=2.0 if not so_um else 1.1, alpha=1.0 if not so_um else 0.5)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color="0.3", ls="-", lw=2),
          Line2D([], [], color="0.3", ls="--", lw=2)]
    l += ["Chrastil", "del Valle & Aguilera"]
    ax.legend(h, l, fontsize=9, ncol=2)
    ax.set_xlabel("Temperatura  [°C]", fontsize=11)
    ax.set_ylabel("Solubilidade  [g extrato / L CO$_2$]", fontsize=11)
    ax.set_title("Solubilidade em CO$_2$ supercrítico", fontsize=13,
                 fontweight="bold")
    ax.grid(alpha=0.3, ls=":")
    n_um = sum(1 for P in pressoes if sum(1 for e in exp if e.P_bar == P) == 1)
    if n_um:
        fig.text(0.5, 0.02,
                 f"linhas tênues: {n_um} isóbaras com um único ponto experimental — "
                 f"a curva ali é extrapolação do modelo global, não ajuste local",
                 ha="center", fontsize=8.5, color="0.35", style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _fechar(pdf, fig)

    # --- paridade ---
    fig, ax = plt.subplots(figsize=(A4[0] - 1.5, A4[0] - 1.5))
    S = np.array([e.S_gL for e in exp])
    lo, hi = 0.75 * S.min(), 1.3 * S.max()
    reta = np.array([lo, hi])
    ax.plot(reta, reta, "-", color="black", lw=1.2, label="y = x")
    ax.fill_between(reta, 0.9 * reta, 1.1 * reta, color="0.6", alpha=0.18,
                    label="±10 %")
    for aj, m, cor in zip(ajs, ("o", "s"), ("#1f77b4", "#d62728")):
        ax.plot(S, aj.S_calc, m, color=cor, ms=9, mec="black", mew=0.7,
                alpha=0.9, label=f"{aj.modelo.split(' (')[0]} (AARD {aj.aard:.1f} %)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("$S_{exp}$  [g/L]"); ax.set_ylabel("$S_{calc}$  [g/L]")
    ax.set_title("Paridade", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, ls=":"); ax.legend(fontsize=9)
    fig.tight_layout()
    _fechar(pdf, fig)


def pagina_conclusoes(pdf: PdfPages, res: list[dict], exp: list, ajs: list) -> None:
    fig, y = _pagina(pdf, "5. Como ler estes números",
                     "o que os dados sustentam e o que não sustentam")

    nao_det = [(r["nome"], p) for r in res for p, (lo, hi) in r["dispersao"].items()
               if lo > 0 and np.log10(hi / lo) > 2]
    total = sum(len(r["dispersao"]) for r in res)

    y = _texto(fig, y, "Cinética", negrito=True, tam=10.5)
    y = _texto(fig, y,
        f"De {total} parâmetros estimados, {len(nao_det)} não são determinados pelos dados:\n"
        f"soluções com DRM equivalente cobrem mais de duas ordens de grandeza. Para esses,\n"
        f"reporte faixa ou limite, nunca um valor pontual.\n\n"
        + ("Não determinados: "
           + ", ".join(f"{n}.{p}" for n, p in nao_det) + "\n\n" if nao_det else "")
        + "DRM baixo e R² alto não validam o ajuste. As condições com melhor DRM desta série\n"
          "incluem parâmetros que variam ordens de grandeza sem efeito mensurável na curva.",
        tam=8.6)

    y = _texto(fig, y, "Solubilidade", negrito=True, tam=10.5)
    ruins = [e.nome for e in exp if not e.saturacao_ok]
    melhor = min(ajs, key=lambda a: a.aard)
    y = _texto(fig, y,
        (f"A premissa de saturação falha em {len(ruins)} de {len(exp)} condições "
         f"({', '.join(ruins)}):\n"
         f"ali o último ponto subestima a solubilidade, e o valor puxa a curva para baixo.\n\n"
         if ruins else "A premissa de saturação se sustenta em todas as condições.\n\n")
        + f"O {melhor.modelo.split(' (')[0]} tem o menor AARD, mas compare os graus de liberdade\n"
          f"antes de concluir: um modelo com mais parâmetros ajusta melhor por construção.\n"
          f"Chrastil tem {ajs[0].dof} e del Valle & Aguilera tem {ajs[1].dof} grau(s) de liberdade.",
        tam=8.6)

    P_um = sorted(P for P in {e.P_bar for e in exp}
                  if sum(1 for e in exp if e.P_bar == P) == 1)
    if P_um:
        y = _texto(fig, y, "Limitação do delineamento", negrito=True, tam=10.5)
        y = _texto(fig, y,
            f"As pressões {P_um} têm uma única temperatura medida. Nelas o efeito da\n"
            f"temperatura não é medido: a isóbara do gráfico é predição do modelo global.\n"
            f"Duas ou três temperaturas por pressão transformariam esse gráfico de\n"
            f"ilustrativo em evidência.", tam=8.6)

    _texto(fig, y,
        "Gerado por: ferramenta de modelagem de EFS — "
        f"{date.today().strftime('%d/%m/%Y')}\n"
        "Dados e configuração em entrada.py; tabelas completas nos CSVs da mesma pasta.",
        tam=7.5, cor="0.5")
    _fechar(pdf, fig)


# =============================================================================
# Programa
# =============================================================================


def coletar():
    """Refaz os ajustes e devolve tudo o que o relatorio precisa."""
    curvas = pipeline.preparar()
    res = []
    for nome, c in curvas.items():
        aj_s, yr = pipeline.ajustar_sovova(c, ent.N_PARTIDAS)
        aj_e = pipeline.ajustar_esquivel(c, ent.N_PARTIDAS)
        g = grupos_sovova(
            Kfa0_1_s=aj_s.valor("Kfa0"), Ksa0_1_s=aj_s.valor("Ksa0"),
            sa=aj_s.valor("sa"), yr_kg_kg=yr, x0_kg_kg=c.x0, N_kg=c.N_kg,
            eps=c.eps, rho_co2_kg_m3=c.rho_co2,
            rho_solido_kg_m3=c.rho_solido, Q_kg_s=c.Q_kg_s)
        avisos = [f"{nome}: {p} mal determinado ({aj_s.dispersao[p][0]:.3g} a "
                  f"{aj_s.dispersao[p][1]:.3g})" for p in aj_s.mal_determinados()]
        avisos += [f"{nome}: {L}" for L in aj_s.nos_limites + aj_e.nos_limites]
        if esquivel_degenerou(aj_e.valor("b"), c.t_s.max()):
            avisos.append(f"{nome}: b > t_máx, o modelo de Esquível degenerou em reta")
        res.append(dict(
            nome=nome, P=c.P_bar, T=c.T_C, Kfa0=aj_s.valor("Kfa0"),
            Ksa0=aj_s.valor("Ksa0"), sa=aj_s.valor("sa"), yr=yr,
            drm_s=aj_s.drm, r2_s=aj_s.r2, e_lim=aj_e.valor("e_lim"),
            b=aj_e.valor("b"), drm_e=aj_e.drm, r2_e=aj_e.r2,
            Z=g.Z, W=g.W, qm=g.qm, qn=g.qn,
            q_min=c.q.min(), q_max=c.q.max(), t_max=c.t_min.max(),
            periodos=descrever_periodos(c.q, g.qm, g.qn).split(":")[0][:10],
            dispersao=aj_s.dispersao, avisos=avisos))

    glob = None
    if getattr(ent, "AJUSTE_GLOBAL_SA", True) and len(curvas) > 1:
        aj_g, sa_g, por = pipeline.ajustar_global_sa(
            curvas, max(24, ent.N_PARTIDAS // 2))
        glob = dict(sa=sa_g, drm=float(np.mean([v["drm"] for v in por.values()])),
                    por_curva=por)
    return curvas, res, glob


def main() -> None:
    pasta = pipeline.pasta_saida()
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "relatorio.pdf"

    print("Gerando relatório em PDF...")
    curvas, res, glob = coletar()
    exp = sol.calcular_experimental()
    ajs = [sol.ajustar_modelo("chrastil", exp), sol.ajustar_modelo("dva", exp)]

    with PdfPages(destino) as pdf:
        pagina_capa(pdf, curvas)
        pagina_config(pdf, curvas)
        pagina_cinetica(pdf, res)
        pagina_diagnostico(pdf, res, glob)
        pagina_solubilidade(pdf, exp, ajs)
        pagina_todas_curvas(pdf, curvas)
        paginas_curvas(pdf, curvas, res)
        paginas_solubilidade(pdf, exp, ajs)
        pagina_conclusoes(pdf, res, exp, ajs)

        d = pdf.infodict()
        d["Title"] = f"EFS com CO2 — {ent.MATERIA_PRIMA}"
        d["Subject"] = "Modelagem cinética e de solubilidade"
        d["Keywords"] = "Sovova, Esquivel, Chrastil, del Valle, CO2 supercritico"

    n_pag = 6 + len(curvas) + 2
    print(f"  {destino}   ({n_pag} páginas)")


if __name__ == "__main__":
    main()
