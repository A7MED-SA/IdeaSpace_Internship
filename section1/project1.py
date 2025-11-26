import numpy as np

def Score_grades(student)->int:
    score=0
    for grade in student:
        score+=grade
    return score



num_students = int(input("Enter the number of students: ")) 
num_grades = int(input("Enter the number of grades: ")) 

for i in range(num_students):
    student = np.array([])
    print("Enter the grades for student",i+1)
    for j in range(num_grades):
        grade = int(input(f"Enter grade {j+1}: "))
        student=np.append(student,grade)
    # print(Score_grades(student))
    score =Score_grades(student)
    if score>=70:
        print(f"Pass : {score}")
    else:
        print(f"Fail : {score}")