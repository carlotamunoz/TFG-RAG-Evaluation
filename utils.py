from collections import defaultdict
import pandas as pd
import ast

def crear_bloques_por_paginas(elements, max_paginas_por_bloque=200):
    paginas = defaultdict(list)
    for el in elements:
        if hasattr(el.metadata, "page_number") and el.metadata.page_number is not None:
            paginas[el.metadata.page_number].append(el.text.strip())
    paginas_ordenadas = sorted(paginas.items())
    bloques = []
    bloque_actual = []
    paginas_actual = 0

    for page_number, textos in paginas_ordenadas:
        bloque_actual.extend(textos)
        paginas_actual += 1
        if paginas_actual == max_paginas_por_bloque:
            bloques.append(bloque_actual)
            bloque_actual = []
            paginas_actual = 0
    if bloque_actual:
        bloques.append(bloque_actual)
    return bloques


def parse_reference_contexts(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or str(val).strip() == "":
        return []
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                res = ast.literal_eval(val)
                if isinstance(res, list):
                    return res
                else:
                    return [res]
            except Exception as e:
                print(f"[WARN] Error al parsear reference_contexts: {val} | Error: {e}")
                return [val]
        else:
            return [val]
    return [val]


# FUNCIÓN para saber si una lista es vacía o contiene solo strings vacíos
def contexto_no_vacio(lista):
        if not lista:
            return False
        if all((x is None) or (isinstance(x, str) and x.strip() == '') for x in lista):
            return False
        return True