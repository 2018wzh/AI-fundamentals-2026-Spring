#include <algorithm>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
const int dst = 123456789;
int src = 0;
const int mov[4] = {3, -3, 1, -1};
const char dir[4] = {'u', 'd', 'l', 'r'};
;
unordered_map<int, int> dis;
unordered_map<int, pair<int, char>> pre;
const int pow10[9] = {1,      10,      100,      1000,     10000,
                      100000, 1000000, 10000000, 100000000};

int h(int s) {
  int res = 0;
  for (int i = 0; i < 9; i++) {
    int v = s / pow10[i] % 10;
    if (v == 9)
      continue;
    int cur_r = (8 - i) / 3, cur_c = (8 - i) % 3;
    int tar_r = (v - 1) / 3, tar_c = (v - 1) % 3;
    res += abs(cur_r - tar_r) + abs(cur_c - tar_c);
  }
  return res;
}

struct Node {
  int s, p, f, g;
  bool operator>(const Node &b) const { return f > b.f; }
};

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  int p = -1;
  for (int i = 0; i < 9; i++) {
    char c;
    cin >> c;
    int pos = 8 - i;
    if (c == 'x') {
      p = pos;
      src += 9 * pow10[pos];
    } else {
      src += (c - '0') * pow10[pos];
    }
  }
  if (p == -1) {
    return -1;
  }
  priority_queue<Node, vector<Node>, greater<Node>> q;
  q.push({src, p, h(src), 0});
  dis[src] = 0;
  while (!q.empty()) {
    Node cur = q.top();
    q.pop();
    int s = cur.s;
    int np = cur.p;
    if (cur.g > dis[s])
      continue;
    if (s == dst) {
      string res = "";
      int res_s = s;
      while (res_s != src) {
        res += pre[res_s].second;
        res_s = pre[res_s].first;
      }
      reverse(res.begin(), res.end());
      cout << res;
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
      if (dis.find(ns) == dis.end() || dis[ns] > cur.g + 1) {
        dis[ns] = cur.g + 1;
        pre[ns] = {s, dir[i]};
        q.push({ns, nnp, dis[ns] + h(ns), dis[ns]});
      }
    }
  }
  return 0;
}