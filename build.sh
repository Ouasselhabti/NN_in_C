#!/bin/BASH

g++ -Wextra  add_gate_nn.c -o out
g++ -Wextra  and_gate_nn.c -o outAnd
echo "------------ THE OR GATE-----------------------"
./out

echo "------------NOW THE AND GATE--------------------"

./outAnd
