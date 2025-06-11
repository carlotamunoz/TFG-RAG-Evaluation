from collections import defaultdict

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
