import numpy as np

student_grades = np.array([56, 79, 59,67,93, 78])
print(student_grades)
class_average = np.average(student_grades)
print(class_average)
highest_grade = np.amax(student_grades)
lowest_grade = np.amin(student_grades)
sorted_grade = np.sort(student_grades)
reveresed_sorted_grades = -np.sort(-student_grades)
print(highest_grade)
print(lowest_grade)
print(sorted_grade)
print(reveresed_sorted_grades)