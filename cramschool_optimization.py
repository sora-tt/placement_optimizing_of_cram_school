import amplify
import numpy as np
import pandas as pd

num_student = 25 # 生徒数
num_teacher = 20 # 講師数
num_subject = 5 # 教科数

# 各講師の教務可能科目情報を乱数で生成
np_data_teacher = np.zeros((num_teacher, num_subject), dtype = int)
for i in range(num_teacher):
  num_enable_subjects = np.random.randint(1, num_subject + 1) # 講師iの教務可能科目数(1～科目数)
  j = 0
  while j < num_enable_subjects:
    k = np.random.randint(num_subject) # 教務可能とする科目の乱数を生成(0～科目数 - 1)
    if np_data_teacher[i][k] == 1: # すでに教務可能となっている場所の上書きは行わず新たな乱数を生成させる
      continue
    np_data_teacher[i][k] = 1 # 講師iは科目kを教務可能とする
    j = j + 1

np_data_teacher = np.array(np_data_teacher)

# 各生徒の授業情報を乱数で生成
np_data_student = np.zeros((num_student, num_subject), dtype = int)
for i in range(num_student):
  j = np.random.randint(num_subject)
  np_data_student[i][j] = 1

# numpy -> pandas
sub_list = [] # 教科リスト
for i in range(num_subject):
  sub_list.append(f"Subject {i}")
df_data_teacher = pd.DataFrame(data = np_data_teacher, columns = sub_list)
df_data_student = pd.DataFrame(data = np_data_student, columns = sub_list)

# print(f"df_data_teacher =\n{df_data_teacher}\n")
# print(f"df_data_student =\n{df_data_student}\n")

# pandas -> numpy
np_data_student = df_data_student.values
np_data_teacher = df_data_teacher.values

print(f"data_teacher =\n{np_data_teacher}\n")
print(f"data_student =\n{np_data_student}\n")

from amplify import BinaryPoly, SymbolGenerator, sum_poly, pair_sum
from amplify.constraint import equal_to, less_equal

gen = SymbolGenerator(BinaryPoly)
L = gen.array(num_teacher, num_student)

student_constraint = sum([equal_to(sum_poly(num_teacher, lambda i: L[i, j]), 1) for j in range(num_student)]) # 生徒1人に講師1人
teacher_constraint = sum([less_equal(sum_poly(num_student, lambda j: L[i, j]), 2) for i in range(num_teacher)]) # 講師1人に生徒2人まで

np_data_product = np_data_teacher @ np_data_student.T

# 講師iが担当できない生徒jのリスト
not_list = []
for i in range(num_teacher):
  for j in range(num_student):
    if np_data_product[i, j] == 0:
      not_list.append([i, j])

common_constraint = sum([equal_to(sum_poly(len(not_list), lambda i: L[not_list[i][0], not_list[i][1]]), 0)]) # 教務可能科目を超える授業不可

constraints = student_constraint + teacher_constraint + common_constraint

inv_num_teacher = 1 / num_teacher
ave = inv_num_teacher * (sum_poly(num_teacher, lambda i: (sum_poly(num_student, lambda j: L[i, j])))) # 各行の和の平均

# var_1 = inv_num_teacher * (sum_poly(num_teacher, lambda i: (sum_poly(num_student, lambda j: L[i, j]) ** 2))) - (inv_num_teacher * ave) ** 2 # 各行の和の分散(最小化問題にするため負にする)
# var_2 = inv_num_teacher * (sum_poly(num_teacher, lambda i: (sum_poly(num_student, lambda j: L[i, j]) - ave) ** 2))
# var_1 = -var_1
# var_2 = -var_2

func_switch = 2 # 目的関数スイッチ
if func_switch == 1:
  var = inv_num_teacher * (sum_poly(num_teacher, lambda i: (sum_poly(num_student, lambda j: L[i, j]) ** 2))) - ave ** 2 # 各行の和の分散1
else:
  var = inv_num_teacher * (sum_poly(num_teacher, lambda i: (sum_poly(num_student, lambda j: L[i, j]) - ave) ** 2)) # 各行の和の分散2
var = -var # 最大化->最小化とするため負に

from amplify.client import FixstarsClient

# クライアントの設定
client = FixstarsClient()
client.token = "xxxxxxxxxxxxxxxx"  # ローカル環境では Amplify AEのアクセストークンを入力してください
client.parameters.timeout = 1000  #  タイムアウト１秒

from amplify import Solver

# 最適化モデル
model = var + constraints

# ソルバーを定義して実行
solver = Solver(client)
result = solver.solve(model)

# 制約条件チェック
if len(result) == 0:
    raise RuntimeError("The given constraints are not satisfied")
values = result[0].values
energy = result[0].energy

# 変数の解
solutions = L.decode(values, 0)
print(f"placement solution:\n{solutions}\n")

teacher_ng_list = [] # 講師制約に違反する解
num_not_commute_teacher = 0 # 非出勤講師数
for i in range(num_teacher):
  res = 0
  for j in range(num_student):
    res = res + solutions[i][j]
  if res < 0 or res > 2:
    teacher_ng_list.append(i)
  if res == 0:
    num_not_commute_teacher = num_not_commute_teacher + 1
print(f"講師制約に違反: {teacher_ng_list}")

student_ng_list = [] # 生徒制約に違反する解
for j in range(num_student):
  res = 0
  for i in range(num_teacher):
    res = res + solutions[i][j]
  if res != 1:
    student_ng_list.append(j)
print(f"生徒制約に違反: {student_ng_list}")

common_ng_list = [] # 生徒講師共通制約に違反する解
for k in range(len(not_list)):
  i = not_list[k][0]
  j = not_list[k][1]
  if solutions[i][j] != 0:
    common_ng_list.append([i, j])
print(f"生徒講師共通制約に違反: {common_ng_list}")

# 配置効率の計算
num_needed_teacher = num_student / 2 # 必要講師数
num_actual_teacher = num_teacher - num_not_commute_teacher # 出勤講師数

placement_efficiency = (num_actual_teacher / num_needed_teacher) * 100 # 配置効率(100%～200%)
print(f"配置効率: {placement_efficiency:.2f}%")
