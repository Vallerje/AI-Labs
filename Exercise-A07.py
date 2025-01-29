import random
class cards:
    def __init__(self):
        self.cards = [] 

    def create(self):
        self.cards = []
        suits = ["s", "h", "d", "c"]
        values = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "A"]
        for suit in suits:
            for value in values:
                self.cards.append(f"{value}{suit}")

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_hands, cards_per_hand):

        hands = [[] for _ in range(num_hands)]
        for _ in range(cards_per_hand):
            for i in range(num_hands):
                if self.cards:  # Check if deck still has cards
                    card = self.cards.pop(0)
                    hands[i].append(card)
                else:
                    break
        return hands

#initialte your programs with this functions
card_01 = cards()  
card_01.create()   
card_01.shuffle() 
dealt_hands = card_01.deal(3, 5)  # Deal 3 hands with 5 cards each

print("Dealt hands:", dealt_hands)
print() 
print("Remaining deck:", card_01.cards)