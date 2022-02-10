import collections
from random import choice

Card = collections.namedtuple("Card", ["rank", "suit"])


class FrenchDesk:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, item):
        print(f"{item}")
        return self._cards.__getitem__(item)


def main():
    v = FrenchDesk()
    Card(1, "") in v


if __name__ == "__main__":
    main()