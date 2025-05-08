import random

def mutate_uniform(chromosome, position, a=-20, b=20):
    """
    Mutacja równomierna - zastępuje gen losową wartością z przedziału [a, b]

    Args:
        chromosome: Chromosom do mutacji
        position: Pozycja genu do zmutowania
        a: Dolna granica przedziału
        b: Górna granica przedziału
    """
    if 0 <= position < len(chromosome):
        # Losowa wartość z dopuszczalnego przedziału
        chromosome[position] = random.uniform(a, b)
    else:
        print(f"Ostrzeżenie: Pozycja {position} poza zakresem chromosomu")

    return chromosome


def mutate_gaussian(chromosome, position, sigma=1.0, a=-20, b=20):
    """
    Mutacja Gaussa - dodaje do genu losową wartość z rozkładu normalnego

    Args:
        chromosome: Chromosom do mutacji
        position: Pozycja genu do zmutowania
        sigma: Odchylenie standardowe dla rozkładu normalnego
        a: Dolna granica przedziału
        b: Górna granica przedziału
    """
    if 0 <= position < len(chromosome):
        # Dodaj losową wartość z rozkładu normalnego
        mutation = random.gauss(0, sigma)
        chromosome[position] += mutation

        # Upewnij się, że wartość pozostaje w granicach
        chromosome[position] = max(a, min(b, chromosome[position]))
    else:
        print(f"Ostrzeżenie: Pozycja {position} poza zakresem chromosomu")

    return chromosome