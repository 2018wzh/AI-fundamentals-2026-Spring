#include <iostream>
#include <queue>
using namespace std;
const int MAXN = 500;
const int INF = 1e9;
int n, m;
struct Edge {
  int v, w;
};
vector<Edge> mp[MAXN + 10];
int dis[MAXN + 10];
int ans = INF;
int main() {
  ios::sync_with_stdio(0);
  cin >> n >> m;
  while (m--) {
    int u, v, w;
    cin >> u >> v >> w;
    mp[u].push_back({v, w});
  }
  for (int i = 1; i <= n; i++)
    dis[i] = INF;
  queue<Edge> q;
  q.push({1, 0});
  dis[1] = 0;
  while (!q.empty()) {
    Edge e = q.front();
    q.pop();
    int u = e.v, w = e.w;
    if (u == n) {
      ans = min(ans, w);
    }
    for (Edge ne : mp[u]) {
      int v = ne.v, nw = w + ne.w;
      if (nw < dis[v]) {
        dis[v] = nw;
        q.push({v, nw});
      }
    }
  }
  if (ans == INF)
    ans = -1;
  cout << ans << endl;
}