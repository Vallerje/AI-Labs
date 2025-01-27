import random

card_suits = ["s", "h", "d", "c"]
card_values = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "A"]
card_deck = []

def createDeck():
    for suit in card_suits:
        for value in card_values:
            card = suit + value

            card_deck.append(card)
    return card_deck

def shuffle(card_deck):
    deck_before_shuffle = card_deck.copy()

    for i in range(len(card_deck)):
            j = random.randint(0, len(card_deck) - 1)
            card_deck[i], card_deck[j] = card_deck[j], card_deck[i]
    return deck_before_shuffle, card_deck


createDeck()
print(card_deck)
shuffle(card_deck)
print("After shuffling:")
print(card_deck)