#!/bin/python
import ast

NUM_SIM_INSTANCE = 20

contents = []

for i in range(NUM_SIM_INSTANCE):
  with open("result"+str(i), "r") as f:
    contents += [f.readlines()]
    
safety_v = []
yield_v = []
turn_v = []
steps = []

for i in range(len(contents[0])):
  cnt= 0
  if i % 4 == 0:
    for j in range(NUM_SIM_INSTANCE):
      cnt += sum(ast.literal_eval(contents[j][i][18:]))
    safety_v += [cnt]
  if i % 4 == 1:
    for j in range(NUM_SIM_INSTANCE):
      cnt += sum(ast.literal_eval(contents[j][i][30:]))
    yield_v += [cnt]
  if i % 4 == 2:
    for j in range(NUM_SIM_INSTANCE):
      cnt += sum(ast.literal_eval(contents[j][i][29:]))
    turn_v += [cnt]
  if i % 4 == 2:
    for j in range(NUM_SIM_INSTANCE):
      cnt += sum(ast.literal_eval(contents[j][i][29:]))
    steps += [cnt]

print(safety_v)
print(yield_v)
print(turn_v)
print(steps)