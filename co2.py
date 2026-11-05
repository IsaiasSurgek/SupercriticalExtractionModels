"""Densidade do CO2: da tabela do usuário, do CoolProp, ou dos dois.

A fonte é escolhida em ``entrada.py``, no campo ``FONTE_DENSIDADE``:

===============  ==========================================================
``"tabela"``     só a tabela; ignora o CoolProp mesmo instalado
``"coolprop"``   só o CoolProp (Span-Wagner); ignora a tabela
``"auto"``       tabela primeiro, CoolProp para o que faltar (padrão)
===============  ==========================================================

Em nenhum modo o programa inventa um valor. Se não souber a densidade, ele
para e explica como resolver. Isso é deliberado: estimar densidade de CO2 perto
do ponto crítico com equação cúbica erra de 10 a 20 %, e é exatamente ali que a
extração supercrítica opera. Valor errado silencioso é pior que programa que
para.

Quanto isso importa
-------------------
A densidade entra no modelo num lugar só, o grupo adimensional

    Z = Kfa0 * rho_CO2 * N / ((1 - eps) * rho_s * Q)

ou seja, sempre no produto ``Kfa0 * rho_CO2``. Os dados não separam os dois.
Um erro de 5 % na densidade vira um erro de 5 % em ``Kfa0`` e **nenhuma**
mudança em Z, no DRM, no ajuste ou em qualquer conclusão de identificabilidade.
Precisão de 1 % é folgada para o que se faz aqui.
"""

from __future__ import annotations

_JA_AVISOU = False

MODOS = ("auto", "tabela", "coolprop")


def coolprop_disponivel() -> bool:
    """``True`` se o pacote CoolProp puder ser importado."""
    try:
        import CoolProp  # noqa: F401
    except ImportError:
        return False
    return True


def _do_coolprop(P_bar: float, T_C: float) -> float | None:
    """Densidade [kg/m3] via CoolProp, ou ``None`` se indisponível/inválido."""
    try:
        from CoolProp.CoolProp import PropsSI
    except ImportError:
        return None
    try:
        return float(PropsSI("D", "P", P_bar * 1e5, "T", T_C + 273.15, "CO2"))
    except (ValueError, RuntimeError):
        return None


def descrever_fonte(modo: str, tabela: dict) -> str:
    """Frase para o cabeçalho do relatório."""
    tem = coolprop_disponivel()
    n = len(tabela)
    if modo == "tabela":
        return f"tabela manual, {n} condicoes (CoolProp desativado por opcao)"
    if modo == "coolprop":
        return ("CoolProp / Span-Wagner" if tem
                else "CoolProp PEDIDO MAS NAO INSTALADO")
    if tem:
        return f"auto: tabela ({n} condicoes) + CoolProp para o resto"
    return f"auto: tabela ({n} condicoes); CoolProp nao instalado"


def _erro(P_bar: float, T_C: float, tabela: dict, modo: str) -> str:
    chave = (int(round(P_bar)), int(round(T_C)))
    disp = ", ".join(f"{p}/{t}" for p, t in sorted(tabela)) or "(tabela vazia)"

    if modo == "coolprop":
        return (
            f"\nERRO: FONTE_DENSIDADE = \"coolprop\", mas o CoolProp nao esta "
            f"instalado.\n"
            f"\nResolva de uma das formas:\n"
            f"      pip install CoolProp\n"
            f"  ou mude em entrada.py:  FONTE_DENSIDADE = \"auto\"\n"
        )

    sugestao_coolprop = (
        ""
        if modo == "tabela"
        else "  1) Instalar o CoolProp, e nunca mais se preocupar com isso:\n"
             "         pip install CoolProp\n\n"
    )
    n = "2" if sugestao_coolprop else "1"
    return (
        f"\nERRO: nao sei a densidade do CO2 a {P_bar:.0f} bar / {T_C:.0f} C.\n"
        f"\n{sugestao_coolprop}"
        f"  {n}) Informar o valor a mao em entrada.py, no bloco DENSIDADE_CO2:\n"
        f"         DENSIDADE_CO2 = {{\n"
        f"             ...\n"
        f"             ({chave[0]}, {chave[1]}): <densidade em kg/m3>,\n"
        f"         }}\n"
        f"     Consulte em https://webbook.nist.gov/chemistry/fluid/\n"
        f"       - escolha Carbon dioxide, unidades SI\n"
        f"       - 'Isothermal properties', T = {T_C:.0f} C, P = {P_bar:.0f} bar\n"
        f"       - leia a coluna Density (kg/m3)\n"
        f"\nCondicoes ja na tabela (bar/C): {disp}\n"
    )


def densidade_co2(
    P_bar: float,
    T_C: float,
    tabela: dict[tuple[int, int], float],
    modo: str = "auto",
) -> tuple[float, str]:
    """Devolve ``(densidade [kg/m3], origem)``.

    Parameters
    ----------
    P_bar, T_C : float
        Condição de operação.
    tabela : dict
        ``DENSIDADE_CO2`` de ``entrada.py``.
    modo : {"auto", "tabela", "coolprop"}
        Ver o topo do módulo.

    Raises
    ------
    SystemExit
        Quando a densidade não pode ser obtida no modo pedido; a mensagem diz
        exatamente o que fazer.
    """
    global _JA_AVISOU

    if modo not in MODOS:
        raise SystemExit(
            f"ERRO: FONTE_DENSIDADE = {modo!r} nao existe. "
            f"Use um de: {', '.join(MODOS)}."
        )

    chave = (int(round(P_bar)), int(round(T_C)))

    if modo in ("auto", "tabela") and chave in tabela:
        return float(tabela[chave]), "tabela"

    if modo == "tabela":
        raise SystemExit(_erro(P_bar, T_C, tabela, modo))

    rho = _do_coolprop(P_bar, T_C)
    if rho is None:
        raise SystemExit(_erro(P_bar, T_C, tabela, modo))

    if not _JA_AVISOU:
        print("  (densidades obtidas do CoolProp / Span-Wagner)")
        _JA_AVISOU = True
    return rho, "CoolProp"
