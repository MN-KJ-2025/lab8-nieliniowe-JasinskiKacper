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
    if not callable(f) or not isinstance(epsilon, float):
        return None
    if not isinstance(max_iter, int):
        return None
    
    fa = f(a)
    fb = f(b)

    if abs(fa) < epsilon:
        return (a, 0)
    if abs(fb) < epsilon:
        return (b, 0)
    if fa * fb > 0:
        return None

    iter = 0
    while iter < max_iter:
        c = (a + b) / 2

        fc = f(c)
        if abs(fc) < epsilon:
            return (c, iter + 1)
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        
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
    if not callable(f):
        return None
    if not isinstance(epsilon, float) or not isinstance(max_iters, (int)):
        return None
    
    x0 = a
    x1 = b
    fx0 = f(x0)
    fx1 = f(x1)

    if abs(fx0) < epsilon:
        return (x0, 0)
    if abs(fx1) < epsilon:
        return (x1, 0)
    
    for iter in range(1, max_iters + 1):
        if fx1 - fx0 == 0:
            return None
        x2 = (x0 * fx1 - x1 * fx0) / (fx1 - fx0)
        fx2 = f(x2)

        if abs(x2 - x1) < epsilon or abs(fx2) <= epsilon:
            return (x2, iter)
        
        x0, x1 = x1, x2
        fx0, fx1 = fx1, fx2

    return (x2, max_iters)


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
    if not callable(f) or not isinstance(x, (int, float)) or not isinstance(h, (int, float)):
        return None
    if h == 0:
        return None
    
    return (f(x + h) - f(x) )/ h
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
    if not callable(f) or not callable(df) or not callable(ddf):
        return None
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if not isinstance(epsilon, float) or not isinstance(max_iter, int):
        return None
    
    if f(a) * f(b) > 0:
        return None
    if f(a) == 0:
        return (a, 0)
    if f(b) == 0:
        return (b, 0)

    if f(a) * ddf(a) > 0:
        x0 = a
    elif f(b) * ddf(b) > 0:
        x0 = b
    else:
        return None

    for it in range(1, max_iter + 1):
        if df(x0) == 0:
            return None
        
        x1 = x0 - f(x0) / df(x0)

        if abs(x1 - x0) < epsilon or abs(f(x1)) < epsilon:
            return (x1, it)
        
        x0 = x1
    return (x1, max_iter)
    pass
