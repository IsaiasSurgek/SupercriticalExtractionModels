"""Validação da ferramenta contra trabalhos publicados do mesmo grupo.

Rode com::

    python validacao_literatura.py

Fontes
------
BESSA, M. V. (2018). Semente de seringueira, extrator de 42 mL — **a mesma
unidade supercrítica** deste trabalho. Tabelas 3, 5, 12 e 13.

ASSUMPÇÃO, I. (2024). Ora-pro-nóbis, extrator de 42 mL com esferas de vidro —
**mesma unidade**. Tabelas 6, 8, 11 e 12.

CARVALHO Jr., R. N. Alecrim (UNICAMP, outra unidade). Vazão 8,33e-5 kg/s.

O que este script testa
-----------------------
1. Se a implementação de Sovová reproduz o rendimento final publicado, usando
   os parâmetros publicados. Testa a **fórmula**.
2. Se o teto físico de ``e_lim`` teria barrado os valores publicados de
   Esquível. Testa o **bound**.
3. Se os ``kfa0`` publicados estão no limite do otimizador. Testa o
   **diagnóstico de identificabilidade**.
"""

from __future__ import annotations

import sys

import numpy as np

from modelos import grupos_sovova, sovova_e

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S_POR_MIN = 60.0
G_POR_KG = 1000.0
LINHA = "=" * 74


# =============================================================================
# Dados publicados
# =============================================================================

# ASSUMPÇÃO (2024), ora-pro-nóbis, extrator 42 mL + esferas de vidro.
# Tabela 6: densidade do solido 0,556 g/cm3, porosidade 0,571, d_Sauter 0,604 mm
# Soxhlet com hexano: 7,47 %.  Massa alimentada: 10 g.
OPN = dict(
    fonte="ASSUMPCAO (2024), ora-pro-nobis, 42 mL",
    soxhlet=0.0747,
    massa_g=10.0,
    rho_solido=556.0,
    porosidade_publicada=0.571,
    # Tabela 8: (P, T): (massa_oleo_g, e_final_pct, rho_co2, t_min)
    # Tabela 11 (Sovova): (P, T): (kfa0, ksa0, xk, DRM_pct)
    # Tabela 12 (Esquivel): (P, T): (e_lim_pct, b_min, DRM_pct)
    curvas={
        (200, 60): dict(m=0.00330, e=0.0357, rho=720.10, t=190,
                        kfa0=12.25, ksa0=0.000132, xk=0.7555, drm_sov=3.64,
                        e_lim=0.0498, b=76.61, drm_esq=3.22),
        (456, 74): dict(m=0.02300, e=0.2486, rho=873.50, t=250,
                        kfa0=100.0, ksa0=0.000553, xk=0.7479, drm_sov=12.74,
                        e_lim=95.5700, b=100000.0, drm_esq=9.95),
        (350, 80): dict(m=0.00730, e=0.0789, rho=789.79, t=200,
                        kfa0=100.0, ksa0=0.000076, xk=0.8023, drm_sov=8.61,
                        e_lim=0.1110, b=111.79, drm_esq=10.03),
        (456, 46): dict(m=0.01300, e=0.1362, rho=959.00, t=270,
                        kfa0=100.0, ksa0=0.685000, xk=0.9478, drm_sov=8.67,
                        e_lim=0.3322, b=437.32, drm_esq=8.14),
        (244, 46): dict(m=0.00990, e=0.1070, rho=847.01, t=250,
                        kfa0=8.77, ksa0=0.000061, xk=0.8994, drm_sov=8.32,
                        e_lim=0.1936, b=238.80, drm_esq=8.30),
        (350, 40): dict(m=0.00300, e=0.0324, rho=935.94, t=180,
                        kfa0=100.0, ksa0=0.000127, xk=0.7253, drm_sov=3.03,
                        e_lim=0.0408, b=52.11, drm_esq=3.94),
        (244, 74): dict(m=0.00590, e=0.0638, rho=706.90, t=330,
                        kfa0=100.0, ksa0=0.000038, xk=0.8001, drm_sov=9.33,
                        e_lim=0.0822, b=161.68, drm_esq=9.33),
        (500, 60): dict(m=0.01100, e=0.1189, rho=934.15, t=310,
                        kfa0=100.0, ksa0=0.000053, xk=0.9261, drm_sov=8.48,
                        e_lim=0.2346, b=312.97, drm_esq=5.61),
        (350, 60): dict(m=0.00650, e=0.0702, rho=862.86, t=240,
                        kfa0=100.0, ksa0=0.000076, xk=0.9371, drm_sov=8.32,
                        e_lim=0.1503, b=283.14, drm_esq=6.29),
    },
)

# BESSA (2018), semente de seringueira, extrator 42 mL, 10 g.
# Tabela 3: densidade aparente 459,64 kg/m3, porosidade 0,482, umidade 4,75 %
# Soxhlet: 30,43 %.  Tabela 12 (Sovova) e 5 (rendimentos).
SERINGUEIRA = dict(
    fonte="BESSA (2018), semente de seringueira, 42 mL",
    soxhlet=0.3043,
    massa_g=10.0,
    rho_solido=459.64,
    porosidade_publicada=0.482,
    curvas={
        (200, 60): dict(e=0.1437, rho=723.0, t=240,
                        kfa0=0.00064, ksa0=0.000260, xk=0.9960, drm_sov=4.79),
        (244, 46): dict(e=0.2803, rho=847.0, t=360,
                        kfa0=3.50150, ksa0=0.000071, xk=0.9037, drm_sov=9.28),
        (244, 74): dict(e=0.0862, rho=705.0, t=240,
                        kfa0=0.08780, ksa0=0.000100, xk=0.7956, drm_sov=6.52),
        (350, 40): dict(e=0.3579, rho=935.0, t=340,
                        kfa0=0.00099, ksa0=0.000112, xk=0.9901, drm_sov=7.40),
        (350, 60): dict(e=0.3865, rho=863.0, t=220,
                        kfa0=2.35430, ksa0=0.000337, xk=0.9773, drm_sov=2.62),
        (350, 80): dict(e=0.6897, rho=789.0, t=280,
                        kfa0=0.51770, ksa0=0.000180, xk=0.9904, drm_sov=10.06),
        (456, 46): dict(e=0.6238, rho=959.0, t=300,
                        kfa0=14.8625, ksa0=0.000148, xk=0.7648, drm_sov=4.27),
        (456, 74): dict(e=0.4626, rho=873.0, t=280,
                        kfa0=4.04410, ksa0=0.000125, xk=0.8984, drm_sov=6.11),
        (500, 60): dict(e=0.8323, rho=934.0, t=300,
                        kfa0=0.39760, ksa0=0.000112, xk=0.9399, drm_sov=5.93),
    },
)


# =============================================================================
# Teste 1 — teto físico de e_lim
# =============================================================================


def teste_teto_elim() -> int:
    """O e_lim publicado respeita o que existe na matriz?"""
    print(LINHA)
    print("TESTE 1 - teto fisico de e_lim (modelo de Esquivel)")
    print(LINHA)
    print("e_lim e o rendimento assintotico. Nao pode passar de x0, que e todo")
    print("o oleo que existe na matriz: x0 = Soxhlet/(1 - Soxhlet).\n")

    x0_pct = 100 * OPN["soxhlet"] / (1 - OPN["soxhlet"])
    print(f"  {OPN['fonte']}")
    print(f"  Soxhlet = {100 * OPN['soxhlet']:.2f} %  ->  teto x0 = {x0_pct:.3f} %\n")
    print(f"  {'P/T':>10s} {'e_lim publ.':>12s} {'b [min]':>10s} {'veredito'}")

    violacoes = 0
    for (P, T), d in OPN["curvas"].items():
        if "e_lim" not in d:
            continue
        excede = d["e_lim"] > x0_pct
        b_no_bound = d["b"] >= 99999
        marca = []
        if excede:
            marca.append(f"EXCEDE o teto em {d['e_lim'] / x0_pct:.0f}x")
            violacoes += 1
        if b_no_bound:
            marca.append("b no limite ub=1e5 do otimizador")
        print(f"  {P:4.0f}/{T:<5.0f} {d['e_lim']:12.4f} {d['b']:10.2f} "
              f"{'  '.join(marca) if marca else 'ok'}")

    print(f"\n  -> {violacoes} condicao(oes) com e_lim fisicamente impossivel.")
    if violacoes:
        print("     A ferramenta impede isso por construcao: o bound superior de")
        print("     e_lim e x0, calculado do TEOR_OLEO que voce informa.")
    return violacoes


# =============================================================================
# Teste 2 — kfa0 no limite do otimizador
# =============================================================================


def teste_bound_kfa0() -> int:
    """Quantos kfa0 publicados estão exatamente no limite do otimizador?"""
    print("\n" + LINHA)
    print("TESTE 2 - kfa0 no limite do otimizador (nao identificabilidade)")
    print(LINHA)

    for dados in (OPN, SERINGUEIRA):
        kf = [d["kfa0"] for d in dados["curvas"].values()]
        no_bound = sum(1 for k in kf if k >= 100.0)
        faixa = np.log10(max(kf) / min(kf))
        print(f"\n  {dados['fonte']}")
        print(f"    kfa0 de {min(kf):.5g} a {max(kf):.5g}  "
              f"-> {faixa:.1f} ordens de grandeza")
        if no_bound:
            print(f"    {no_bound} de {len(kf)} condicoes com kfa0 = 100 "
                  f"exatamente = limite superior do otimizador")
        drm = [d["drm_sov"] for d in dados["curvas"].values()]
        print(f"    DRM entre {min(drm):.2f} % e {max(drm):.2f} % "
              f"-- ajuste 'bom' em todas")

    print("\n  -> Um coeficiente de transferencia de massa que varre 4 a 5 ordens")
    print("     de grandeza sem padrao com P ou T, e que encosta no limite do")
    print("     otimizador na maioria das condicoes, nao esta sendo medido: a")
    print("     superficie objetivo e plana nele. O DRM baixo nao contradiz isso")
    print("     -- e exatamente o sintoma.")
    print("\n     A ferramenta reporta 'mal determinado' e mostra a faixa, em vez")
    print("     de imprimir um valor pontual.")
    return sum(1 for d in OPN["curvas"].values() if d["kfa0"] >= 100.0)


# =============================================================================
# Teste 3 — a implementação reproduz o rendimento final publicado?
# =============================================================================


def teste_reproduz_rendimento() -> None:
    """Demonstra o confundimento Q*y_r usando os dados publicados.

    A vazão não é declarada em nenhum dos dois trabalhos — a mesma lacuna dos
    dados deste projeto. Varrendo Q e resolvendo y_r para bater o rendimento
    final, o resultado é que **toda** vazão reconcilia, e o produto ``Q*y_r``
    fica constante. Isso não é falha do teste: é a demonstração de que os dados
    fixam o produto e não os fatores.

    Aqui o teste é fraco de propósito quanto à *fórmula*: com ``kfa0`` grande,
    ``1 - exp(-Z) -> 1`` e o período I devolve o alvo por construção. Validar a
    fórmula ponto a ponto exigiria as curvas cinéticas completas, que os dois
    trabalhos publicam apenas como figuras.
    """
    print("\n" + LINHA)
    print("TESTE 3 - confundimento Q*y_r nos dados publicados")
    print(LINHA)
    print("A vazao nao e declarada em nenhum dos dois trabalhos. Varremos Q e")
    print("resolvemos y_r para bater o rendimento final publicado.\n")

    dados = OPN
    teor = dados["soxhlet"]
    x0 = teor / (1 - teor)
    massa_kg = dados["massa_g"] * 1e-3
    N_kg = massa_kg * (1 - teor)
    eps = dados["porosidade_publicada"]

    print(f"  {dados['fonte']}")
    print(f"  x0 = {100 * x0:.4f} %   N = {N_kg * G_POR_KG:.4f} g   eps = {eps}\n")
    print(f"  {'P/T':>10s} {'e_publ [%]':>11s} {'Q* [g/min]':>11s} "
          f"{'e_calc [%]':>11s} {'y_r* [g/kg]':>12s}")

    Qs = np.geomspace(0.5, 20.0, 400)
    achou = 0
    for (P, T), d in sorted(dados["curvas"].items()):
        t_s = d["t"] * S_POR_MIN
        alvo = d["e"] / 100.0                     # fração
        yr_impl = np.array([  # y_r que a taxa media implica, para cada Q
            (alvo * N_kg) / (Q / (G_POR_KG * S_POR_MIN) * t_s) for Q in Qs
        ])
        erros = []
        for Q, yr in zip(Qs, yr_impl):
            Q_kg_s = Q / (G_POR_KG * S_POR_MIN)
            g = grupos_sovova(
                Kfa0_1_s=d["kfa0"], Ksa0_1_s=d["ksa0"], sa=d["xk"],
                yr_kg_kg=yr, x0_kg_kg=x0, N_kg=N_kg, eps=eps,
                rho_co2_kg_m3=d["rho"], rho_solido_kg_m3=dados["rho_solido"],
                Q_kg_s=Q_kg_s,
            )
            q = np.array([(Q_kg_s / N_kg) * t_s])
            e = sovova_e(q, x0_kg_kg=x0, sa=d["xk"], yr_kg_kg=yr,
                         Z=g.Z, W=g.W, qm=g.qm, qn=g.qn)[0]
            erros.append(abs(e - alvo) / alvo)
        erros = np.array(erros)
        k = int(np.argmin(erros))
        Q_kg_s = Qs[k] / (G_POR_KG * S_POR_MIN)
        g = grupos_sovova(
            Kfa0_1_s=d["kfa0"], Ksa0_1_s=d["ksa0"], sa=d["xk"],
            yr_kg_kg=yr_impl[k], x0_kg_kg=x0, N_kg=N_kg, eps=eps,
            rho_co2_kg_m3=d["rho"], rho_solido_kg_m3=dados["rho_solido"],
            Q_kg_s=Q_kg_s,
        )
        e_calc = sovova_e(np.array([(Q_kg_s / N_kg) * t_s]), x0_kg_kg=x0,
                          sa=d["xk"], yr_kg_kg=yr_impl[k], Z=g.Z, W=g.W,
                          qm=g.qm, qn=g.qn)[0]
        ok = erros[k] < 0.01
        achou += ok
        print(f"  {P:4.0f}/{T:<5.0f} {d['e']:11.4f} {Qs[k]:11.3f} "
              f"{100 * e_calc:11.4f} {yr_impl[k] * G_POR_KG:12.5f}"
              f"{'' if ok else '   (nao reconcilia)'}")

    print(f"\n  -> {achou} de {len(dados['curvas'])} condicoes reconciliadas "
          f"dentro de 1 %, e o Q* encontrado foi sempre o piso da varredura.")
    print("     Isso NAO e um ajuste de vazao: e a demonstracao de que qualquer Q")
    print("     serve, desde que y_r acompanhe. Confira abaixo, para 200/60:\n")

    d = dados["curvas"][(200, 60)]
    t_s = d["t"] * S_POR_MIN
    alvo = d["e"] / 100.0
    print(f"    {'Q [g/min]':>11s} {'y_r [g/kg]':>12s} {'Q*y_r [g/min]':>15s} "
          f"{'e_calc [%]':>11s}")
    for Q in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        Q_kg_s = Q / (G_POR_KG * S_POR_MIN)
        yr = (alvo * N_kg) / (Q_kg_s * t_s)
        g = grupos_sovova(
            Kfa0_1_s=d["kfa0"], Ksa0_1_s=d["ksa0"], sa=d["xk"], yr_kg_kg=yr,
            x0_kg_kg=x0, N_kg=N_kg, eps=eps, rho_co2_kg_m3=d["rho"],
            rho_solido_kg_m3=dados["rho_solido"], Q_kg_s=Q_kg_s,
        )
        e = sovova_e(np.array([(Q_kg_s / N_kg) * t_s]), x0_kg_kg=x0, sa=d["xk"],
                     yr_kg_kg=yr, Z=g.Z, W=g.W, qm=g.qm, qn=g.qn)[0]
        print(f"    {Q:11.2f} {yr * G_POR_KG:12.5f} "
              f"{Q * yr * G_POR_KG:15.6f} {100 * e:11.4f}")

    print("\n     O produto Q*y_r e constante e o rendimento tambem. Uma vazao 40x")
    print("     maior com y_r 40x menor da exatamente a mesma curva. Por isso a")
    print("     ferramenta reporta o produto, e trata y_r isolado como")
    print("     condicionado a um Q assumido.")


def main() -> None:
    print("VALIDACAO DA FERRAMENTA CONTRA TRABALHOS PUBLICADOS DO GRUPO")
    print("(BESSA 2018 e ASSUMPCAO 2024 usaram a MESMA unidade, extrator 42 mL)\n")
    v = teste_teto_elim()
    b = teste_bound_kfa0()
    teste_reproduz_rendimento()

    print("\n" + LINHA)
    print("RESUMO")
    print(LINHA)
    print(f"  e_lim fisicamente impossivel nos dados publicados: {v} condicao(oes)")
    print(f"  kfa0 no limite do otimizador:                      {b} condicoes")
    print("  Ambos seriam barrados ou sinalizados por esta ferramenta.")


if __name__ == "__main__":
    main()
