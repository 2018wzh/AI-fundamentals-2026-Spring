#include <iostream>
#include <queue>
#include <vector>
using namespace std;
const int MAXN = 1e5;
int n, m;
int ans = -1;
vector<int> mp[MAXN + 10];
int vis[MAXN + 10];
int main() {
  ios::sync_with_stdio(0);
  cin >> n >> m;
  while (m--) {
    int u, v;
    cin >> u >> v;
    mp[u].push_back(v);
  }
  queue<int> q;
  q.push(1);
  vis[1] = 1;
  while (!q.empty()) {
    int u = q.front();
    if (u == n) {
      ans = vis[u] - 1;
      break;
    }
    int stp = vis[u];
    q.pop();
    for (int v : mp[u]) {
      if (!vis[v]) {
        vis[v] = stp + 1;
        q.push(v);
      }
    }
  }
  cout << ans;
}