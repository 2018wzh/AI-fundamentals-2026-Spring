#include <iostream>
#include <queue>
#include <vector>
using namespace std;
int m, n;
const int MAXN = 150000;
int dis[MAXN + 10];
const int Inf = 1e9;
typedef long long ll;
struct Edge {
  int v, w;
  bool operator<(const Edge &b) const { return w == b.w ? v > b.v : w > b.w; }
};
priority_queue<Edge> q;
void init(int n) {
  for (int i = 0; i <= n; i++)
    dis[i] = Inf;
}
vector<Edge> M[MAXN + 10];
void Dijkstra(int x) {
  dis[x] = 0;
  q.push({x, 0});
  while (!q.empty()) {
    auto cur = q.top();
    q.pop();
    if (cur.w > dis[cur.v])
      continue;
    for (auto i : M[cur.v])
      if (dis[i.v] > dis[cur.v] + i.w) {
        dis[i.v] = dis[cur.v] + i.w;
        q.push({i.v, dis[i.v]});
      }
  }
}
int main() {
  ios::sync_with_stdio(0);
  cin >> n >> m;
  init(n);
  while (m--) {
    int u, v, w;
    cin >> u >> v >> w;
    M[u].push_back({v, w});
  }
  Dijkstra(1);
  cout << dis[n];
}