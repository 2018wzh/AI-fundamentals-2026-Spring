#include <iostream>
#include <queue>
using namespace std;
int m, n;
const int MAXN = 100;
bool mp[MAXN + 10][MAXN + 10];
int vis[MAXN + 10][MAXN + 10];
typedef pair<int, int> pii;
const int mov[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
inline bool chk(pii p) {
  return p.first >= 0 && p.first < m && p.second >= 0 && p.second < n;
}
int bfs() {
  queue<pii> q;
  q.push({0, 0});
  vis[0][0] = 1;
  int ans = 0;
  while (!q.empty()) {
    pii u = q.front();
    q.pop();
    int stp = vis[u.first][u.second];
    if (u.first == m - 1 && u.second == n - 1)
      return stp;
    for (int i = 0; i < 4; i++) {
      pii v = {u.first + mov[i][0], u.second + mov[i][1]};
      if (chk(v) && vis[v.first][v.second] == 0 && mp[v.first][v.second] == 0) {
        vis[v.first][v.second] = stp + 1;
        q.push(v);
      }
    }
  }
  return 1e9;
}
int dfs(int x = 0, int y = 0, int stp = 1) {
  if (x == m - 1 && y == n - 1)
    return stp;
  int ans = 1e9;
  for (int i = 0; i < 4; i++) {
    pii v = {x + mov[i][0], y + mov[i][1]};
    if (chk(v) && vis[v.first][v.second] == 0 && mp[v.first][v.second] == 0) {
      vis[v.first][v.second] = 1;
      int res = dfs(v.first, v.second, stp + 1);
      ans = min(res, ans);
      vis[v.first][v.second] = 0;
    }
  }
  return ans;
}
int dij() {
  priority_queue<pair<int, pii>, vector<pair<int, pii>>,
                 greater<pair<int, pii>>>
      q;
  q.push({1, {0, 0}});
  vis[0][0] = 1;
  int ans = 1e9;
  while (!q.empty()) {
    auto u = q.top();
    q.pop();
    int stp = u.first;
    pii pos = u.second;
    if (pos.first == m - 1 && pos.second == n - 1)
      ans = min(ans, stp);
    for (int i = 0; i < 4; i++) {
      pii v = {pos.first + mov[i][0], pos.second + mov[i][1]};
      if (chk(v) && vis[v.first][v.second] == 0 && mp[v.first][v.second] == 0) {
        vis[v.first][v.second] = 1;
        q.push({stp + 1, v});
      }
    }
  }
  return ans;
}
int astar() {
  auto h = [](pii p) { return abs(p.first - m + 1) + abs(p.second - n + 1); };
  priority_queue<pair<int, pii>, vector<pair<int, pii>>,
                 greater<pair<int, pii>>>
      q;
  q.push({1 + h({0, 0}), {0, 0}});
  vis[0][0] = 1;
  int ans = 1e9;
  while (!q.empty()) {
    auto u = q.top();
    q.pop();
    int stp = u.first - h(u.second);
    pii pos = u.second;
    if (pos.first == m - 1 && pos.second == n - 1)
      ans = min(ans, stp);
    for (int i = 0; i < 4; i++) {
      pii v = {pos.first + mov[i][0], pos.second + mov[i][1]};
      if (chk(v) && vis[v.first][v.second] == 0 && mp[v.first][v.second] == 0) {
        vis[v.first][v.second] = 1;
        q.push({stp + 1 + h(v), v});
      }
    }
  }
  return ans;
}
int main() {
  ios::sync_with_stdio(0);
  cin >> m >> n;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      cin >> mp[i][j];
  int ans = bfs();
  cout << (ans == 1e9 ? -1 : ans - 1) << endl;
}