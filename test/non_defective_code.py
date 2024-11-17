class TestClass:
    
    def function_with_if_else(self, x):
        """
        A simple function with an if-else statement.
        This will help test Radon's cyclomatic complexity.
        """
        if x > 0:
            return "Positive"
        elif x == 0:
            return "Zero"
        else:
            return "Negative"
    
    def function_with_for_loop(self, n):
        """
        A function that loops over a range of numbers.
        This will add more complexity due to the loop.
        """
        for i in range(n):
            if i % 2 == 0:
                print(f"{i} is even")
            else:
                print(f"{i} is odd")
    
    def function_with_while_loop(self, n):
        """
        A function that uses a while loop and a break condition.
        The loop adds to the cyclomatic complexity.
        """
        i = 0
        while i < n:
            if i == 5:
                break
            i += 1
        return i

    def function_with_nested_conditionals(self, x, y):
        """
        A function with nested conditionals to increase cyclomatic complexity.
        """
        if x > 0:
            if y > 0:
                return "Both positive"
            elif y < 0:
                return "x positive, y negative"
            else:
                return "x positive, y zero"
        elif x < 0:
            return "x negative"
        else:
            return "x zero"

    def function_with_multiple_conditions(self, a, b, c):
        """
        A function with multiple conditions to further increase complexity.
        """
        if a > b:
            if b > c:
                return "a > b > c"
            elif c > b:
                return "a > c > b"
            else:
                return "a > b == c"
        elif a == b:
            return "a == b"
        else:
            return "a < b"
    
    def function_with_switch_case(self, choice):
        """
        A simulated switch-case statement (Python doesn't have a built-in switch-case).
        This will add complexity with multiple choices.
        """
        if choice == 1:
            return "Choice 1"
        elif choice == 2:
            return "Choice 2"
        elif choice == 3:
            return "Choice 3"
        else:
            return "Unknown choice"
    
    def function_with_exception_handling(self, x):
        """
        A function that includes exception handling (try/except).
        This will help check Radon's ability to handle exception blocks.
        """
        try:
            result = 10 / x
        except ZeroDivisionError:
            return "Division by zero error"
        except Exception as e:
            return f"An error occurred: {e}"
        return result

# Example usage
if __name__ == "__main__":
    obj = TestClass()
    print(obj.function_with_if_else(1))  # Expected: Positive
    obj.function_with_for_loop(5)        # Prints numbers 0 to 4 with odd/even
    print(obj.function_with_while_loop(10))  # Expected: 5 (breaks when i == 5)
    print(obj.function_with_nested_conditionals(1, -1))  # Expected: "x positive, y negative"
    print(obj.function_with_multiple_conditions(3, 2, 1))  # Expected: "a > b > c"
    print(obj.function_with_switch_case(2))  # Expected: "Choice 2"
    print(obj.function_with_exception_handling(0))  # Expected: "Division by zero error"
