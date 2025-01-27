import random

def flip_coin():
    if random.randint(0, 1) == 0:
        return 'H'
    else:
        return 'T'

def simulate_flips():
    consecutive_heads = 0
    consecutive_tails = 0
    flips = 0
    outcomes = []

    while consecutive_heads < 3 and consecutive_tails < 3:
        outcome = flip_coin()
        outcomes.append(outcome)
        flips += 1

        if outcome == 'H':
            consecutive_heads += 1
            consecutive_tails = 0
        else:
            consecutive_heads = 0
            consecutive_tails += 1

    print(''.join(outcomes))
    print(f"Number of flips: {flips}")

if __name__ == "__main__":
    simulate_flips()