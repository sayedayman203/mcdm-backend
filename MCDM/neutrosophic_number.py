class NeutrosophicNumber:
    def __init__(self, low, mid, high, T, I, F) -> None:
        self.low = low
        self.mid = mid
        self.high = high
        self.T = T
        self.I = I
        self.F = F

    def de_nutrosophication(self):
        return (
            1 / 8 * (self.low + self.mid + self.high) * (2 + self.T - self.I - self.F)
        )

    def __repr__(self) -> str:
        return f"(({self.low}, {self.mid}, {self.high}), {self.T}, {self.I}, {self.F})"

    def __add__(self, nset):
        return NeutrosophicNumber(
            self.low + nset.low,
            self.mid + nset.mid,
            self.high + nset.high,
            min(self.T, nset.T),
            max(self.I, nset.I),
            max(self.F, nset.F),
        )
