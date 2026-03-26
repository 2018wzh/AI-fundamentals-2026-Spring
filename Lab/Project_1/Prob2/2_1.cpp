#include <iostream>
#include <stack>
#include <unordered_set>
using namespace std;
const int dst = 123456789;
int src = 0;
const int mov[4] = {1, -1, 3, -3};
unordered_set<int> vis;
const int pow10[9] = {1,      10,      100,      1000,     10000,
                      100000, 1000000, 10000000, 100000000};
int main() {
  ios::sync_with_stdio(0);
  int p = -1;
  for (int i = 8; i >= 0; i--) {
    char x;
    cin >> x;
    if (x == 'x') {
      p = i;
      x = '9';
    }
    int t = x - '0';
    src += t * pow10[i];
  }
  if (p == -1)
    return -1;
  stack<pair<int, int>> stk;
  stk.push({src, p});
  while (!stk.empty()) {
    auto [s, np] = stk.top();
    stk.pop();
    if (s == dst) {
      cout << 1;
      return 0;
    }
    vis.insert(s);
    for (int i = 0; i < 4; i++) {
      int nnp = np + mov[i];
      if (nnp < 0 || nnp >= 9 || (mov[i] == 1 && np % 3 == 2) ||
          (mov[i] == -1 && np % 3 == 0))
        continue;
      int a = s / pow10[nnp] % 10;
      int ns =
          s - 9 * pow10[np] + a * pow10[np] - a * pow10[nnp] + 9 * pow10[nnp];
      if (vis.count(ns))
        continue;
      stk.push({ns, nnp});
    }
  }
  cout << 0;
  return 0;
}