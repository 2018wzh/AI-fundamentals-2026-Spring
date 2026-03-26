#include <iostream>
#include <queue>
#include <unordered_set>
#include <unordered_map>
using namespace std;
const int dst = 123456789;
int src = 0;
const int mov[4] = {1, -1, 3, -3};
unordered_map<int, int> dis;
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
  queue<pair<int, int>> q;
  q.push({src, p});
  dis[src] = 0;
  while (!q.empty()) {
    auto cur = q.front();
    int s = cur.first;
    int np = cur.second;
    q.pop();
    if (s == dst) {
      cout << dis[s];
      return 0;
    }
    for (int i = 0; i < 4; i++) {
      int nnp = np + mov[i];
      if (nnp < 0 || nnp >= 9 || (mov[i] == 1 && np % 3 == 2) ||
          (mov[i] == -1 && np % 3 == 0))
        continue;
      int a = s / pow10[nnp] % 10;
      int ns =
          s - 9 * pow10[np] + a * pow10[np] - a * pow10[nnp] + 9 * pow10[nnp];
      if (dis.count(ns))
        continue;
      dis[ns] = dis[s] + 1;
      q.push({ns, nnp});
    }
  }
  cout << -1;
  return 0;
}