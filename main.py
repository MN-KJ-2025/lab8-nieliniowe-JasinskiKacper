# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np
from typing import Callable


def func(x: int | float | np.ndarray) -> int | float | np.ndarray:
    """Funkcja wyliczająca wartości funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.exp(-2 * x) + x**2 - 1


def dfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości pierwszej pochodnej (df(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    df(x) = -2 * e^(-2x) + 2x

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return -2 * np.exp(-2 * x) + 2 * x


def ddfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości drugiej pochodnej (ddf(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    ddf(x) = 4 * e^(-2x) + 2

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return 4 * np.exp(-2 * x) + 2


def bisection(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] 
    metodą bisekcji.

    Args:
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iter (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if not isinstance(f, Callable[[float], float]) or not isinstance(epsilon, float):
        return None
    if not isinstance(max_iter, float):
        return None
    
    if func(a) == 0:
        return (a, 0)
    if func(b) == 0:
        return (b, 0)

    iter = 0
    i = 2
    while func(a) * func(b) < 0:
        if iter >= max_iter:
            break
        
        c = (a + b) / i
        if abs(c - a) < abs(c - b):
            if func(a) * func(c) >= 0:
                break
            else:
                b = c
        else:
            if func(b) * func(c) >= 0:
                break
            else:
                a = c
        i *=2
        iter += 1

    return (c, iter)
    pass


def secant(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iters: int,
) -> tuple[float, int] | None:
    """funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] 
    metodą siecznych.

    Args:
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iters (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if not isinstance(f, Callable[[float], float]):
        return None
    if not isinstance(epsilon, float) or not isinstance(max_iters, (int)):
        return None
    
    if func(a) == 0:
        return (a, 0)
    if func(b) == 0:
        return (b, 0)
    
    iter = 0
    for _ in range(max_iters):
        c = (a*func(b) - b*func(a)) / (func(b) - func(a))
        if func(c) == 0:
            break
    
        iter += 1
    return (c, iter)
    pass


def difference_quotient(
    f: Callable[[float], float], x: int | float, h: int | float
) -> float | None:
    """Funkcja obliczająca wartość iloazu różnicowego w punkcie x dla zadanej 
    funkcji f(x).

    Args:
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        x (int | float): Argument funkcji.
        h (int | float): Krok różnicy wykorzystywanej do wyliczenia ilorazu 
            różnicowego.

    Returns:
        (float): Wartość ilorazu różnicowego.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    pass


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    ddf: Callable[[float], float],
    a: int | float,
    b: int | float,
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.

    Args:
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        df (Callable[[float], float]): Pierwsza pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        ddf (Callable[[float], float]): Druga pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iter (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    pass
