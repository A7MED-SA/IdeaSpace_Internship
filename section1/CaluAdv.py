import operator
import math

class Calculator:
    def __init__(self):
        # Dictionary mapping operations to functions
        self.operations = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
            'sqrt': lambda x, _: math.sqrt(x),
            'sin': lambda x, _: math.sin(x),
            'cos': lambda x, _: math.cos(x),
            'tan': lambda x, _: math.tan(x),
            'log': lambda x, _: math.log(x),
            'abs': lambda x, _: abs(x)
        }
        
        # Error messages for different scenarios
        self.error_messages = {
            ZeroDivisionError: "خطأ: لا يمكن القسمة على صفر",
            ValueError: "خطأ: قيمة غير صحيحة",
            KeyError: "خطأ: عملية غير مدعومة",
            TypeError: "خطأ: نوع البيانات غير صحيح"
        }
    
    def calculate(self, num1, operation, num2=None):
        """
        Calculate result without using if statements
        """
        try:
            # Get the operation function from dictionary
            operation_func = self.operations[operation]
            
            # Execute the operation
            result = operation_func(num1, num2 or 0)
            return result
            
        except Exception as e:
            # Return error message based on exception type
            return self.error_messages.get(type(e), f"خطأ غير معروف: {str(e)}")
    
    def get_input(self):
        """
        Get user input without if statements
        """
        try:
            print("العمليات المتاحة:")
            operations_list = list(self.operations.keys())
            print(", ".join(operations_list))
            
            num1 = float(input("أدخل الرقم الأول: "))
            operation = input("أدخل العملية: ")
            
            # For single operand operations, num2 is not needed
            single_operand_ops = ['sqrt', 'sin', 'cos', 'tan', 'log', 'abs']
            num2 = None
            
            # Use dictionary lookup instead of if
            need_second_number = operation not in single_operand_ops
            num2 = float(input("أدخل الرقم الثاني: ")) * need_second_number + (not need_second_number) * 0
            
            return num1, operation, num2
            
        except ValueError:
            return None, None, None
    
    def run(self):
        """
        Main calculator loop without if statements
        """
        print("مرحباً بك في الآلة الحاسبة المتقدمة!")
        print("اكتب 'exit' للخروج")
        
        while True:
            try:
                # Get input
                num1, operation, num2 = self.get_input()
                
                # Check for exit using dictionary lookup
                exit_commands = {'exit': True, 'خروج': True, 'quit': True}
                should_exit = exit_commands.get(str(operation), False)
                
                # Exit using exception instead of if
                should_exit and (_ for _ in ()).throw(StopIteration)
                
                # Calculate result
                result = self.calculate(num1, operation, num2)
                print(f"النتيجة: {result}")
                print("-" * 30)
                
            except StopIteration:
                print("شكراً لاستخدام الآلة الحاسبة!")
                break
            except Exception as e:
                print(f"خطأ في الإدخال: {str(e)}")
                print("يرجى المحاولة مرة أخرى")
                continue

# Example usage functions without if statements
def demo_calculations():
    """Demonstrate calculator usage"""
    calc = Calculator()
    
    # Test cases using dictionary
    test_cases = {
        "جمع": (10, '+', 5),
        "طرح": (10, '-', 3),
        "ضرب": (4, '*', 7),
        "قسمة": (15, '/', 3),
        "أس": (2, '**', 3),
        "جذر تربيعي": (16, 'sqrt', None),
        "قيمة مطلقة": (-5, 'abs', None)
    }
    
    print("أمثلة على العمليات الحسابية:")
    print("=" * 40)
    
    for description, (num1, op, num2) in test_cases.items():
        result = calc.calculate(num1, op, num2)
        num2_str = f" و {num2}" * bool(num2 is not None)
        print(f"{description}: {num1}{num2_str} = {result}")

# Run the calculator
if __name__ == "__main__":
    # Create calculator instance
    calculator = Calculator()
    
    # Show demo first
    demo_calculations()
    print("\n")
    
    # Run interactive calculator
    calculator.run()