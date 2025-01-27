def prime_number_checker(number):
    if number <= 1:
        return False  
    for i in range(2, int(number ** 0.5) + 1):  # Check divisors up to the square root of the number
        if number % i == 0:
            return False 
    return True  

def main():
    try:
        num = int(input("Enter an integer: "))
        if prime_number_checker(num):
            print(f"{num} is a prime number.")
        else:
            print(f"{num} is not a prime number.")
    except ValueError:
        print("Error: Please enter a valid integer.")

if __name__ == "__main__":
    main()
