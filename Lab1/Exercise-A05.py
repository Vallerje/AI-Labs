def is_prime(number):
    if number <= 1:
        return False  
    for i in range(2, int(number ** 0.5) + 1): 
        if number % i == 0:
            return False  
    return True  

def next_prime(n):
    candidate = n + 1  
    while not is_prime(candidate):  
        candidate += 1
    return candidate

def main():
    try:
        num = int(input("Enter an integer: "))
        next_prime_number = next_prime(num)
        print(f"The first prime number larger than {num} is {next_prime_number}.")
    except ValueError:
        print("Error: Please enter a valid integer.")

# Run the main program
if __name__ == "__main__":
    main()
