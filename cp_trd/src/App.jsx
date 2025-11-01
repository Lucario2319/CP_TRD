import React, { useState } from 'react';
import { Copy, Check, Menu, X, Search } from 'lucide-react';
import CodeBlock from './components/CodeBlock';

const Section = ({ title, children, id }) => {
  return (
    <section id={id} className="mb-12 scroll-mt-20">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 pb-2 border-2 border-blue-300">
        {title}
      </h2>
      {children}
    </section>
  );
};

const SubSection = ({ title, children }) => {
  return (
    <div className="mb-8">
      <h3 className="text-xl font-semibold text-gray-300 mb-4 border-2">{title}</h3>
      {children}
    </div>
  );
};

export default function CPTemplateRepository() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  const sections = [
    { id: 'template', title: 'Main Template' },
    { id: 'bit-manipulation', title: 'Bit Manipulation' },
    { id: 'difference-arrays', title: 'Difference Arrays' },
    { id: 'fenwick-tree', title: 'Fenwick Tree (BIT)' },
    { id: 'graphs', title: 'Graph Algorithms' },
    { id: 'mod-ncr', title: 'Modular Arithmetic & NCR' },
    { id: 'primes', title: 'Prime Numbers' },
    { id: 'priority-queue', title: 'Priority Queue' },
    { id: 'scc', title: 'Strongly Connected Components' },
    { id: 'segment-tree', title: 'Segment Tree' },
    { id: 'cpit-tools', title: 'CPIT Tools' },
  ];

  const filteredSections = sections.filter(section =>
    section.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    setSidebarOpen(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 hover:bg-gray-100 rounded-lg"
            >
              {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
            <h1 className="text-2xl font-bold text-gray-800">
              CP Template Repository
            </h1>
          </div>
          <div className="hidden sm:block text-sm text-gray-600">
            Competitive Programming Templates & Tools
          </div>
        </div>
      </header>

      <div className="flex max-w-7xl mx-auto">
        {/* Sidebar */}
        <aside
          className={`${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          } lg:translate-x-0 fixed lg:sticky top-16 left-0 h-[calc(100vh-4rem)] w-64 bg-white shadow-lg lg:shadow-none border-r border-gray-200 transition-transform duration-300 ease-in-out z-30 overflow-y-auto`}
        >
          <div className="p-4">
            <div className="relative mb-4">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
              <input
                type="text"
                placeholder="Search sections..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <nav>
              <ul className="space-y-2">
                {filteredSections.map((section) => (
                  <li key={section.id}>
                    <button
                      onClick={() => scrollToSection(section.id)}
                      className="w-full text-left px-4 py-2 rounded-lg hover:bg-blue-50 hover:text-blue-600 transition-colors text-gray-700"
                    >
                      {section.title}
                    </button>
                  </li>
                ))}
              </ul>
            </nav>
          </div>
        </aside>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div
            className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-20"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main Content */}
        <main className="flex-1 p-6 lg:p-8">
          {/* Template Section */}
          <Section id="template" title="Main Template">
            <SubSection title="Python Template">
              <CodeBlock
                language="python"
                code={`import sys
from math import log2, ceil, gcd, lcm, comb, perm, factorial
from collections import deque, defaultdict, Counter
from bisect import bisect_left, bisect_right # use key = lambda smth for custom check
from itertools import accumulate

II = lambda: int(input())
SI = lambda: input()
MI = lambda: map(int, input().split())
LI = lambda: list(map(int, input().split()))
P  = lambda *x: print(*x)

def solve():    # to process each test case
    n,q = MI() 
    a = LI()

def main():
    t = II()
    for _ in range(t):
        solve()
        
if __name__ == "__main__":
    try:
        sys.stdin = open("input.txt", "r")
    except FileNotFoundError:
        pass
    main()`}
              />
            </SubSection>
          </Section>

          {/* Bit Manipulation */}
          <Section id="bit-manipulation" title="Bit Manipulation">
            <div className="mb-4 p-4 bg-blue-50 rounded-lg">
              <p className="text-gray-700">
                Essential bit manipulation operations for competitive programming.
              </p>
            </div>
            <CodeBlock
              language="python"
              code={`from math import log2

def find_right_most_one(n):
    m = n & (n ^ (n - 1))
    print("Right most 1 is present at", int(log2(m)), "and value is", m)

def find_left_most_one(n):
    m = 1 << (n.bit_length() - 1)
    print("Left most 1 is present at", int(log2(m)), "and value is", m)

def flip_right_most_one(n):
    m = n & (n - 1)
    print("After flipping right most one, value is", m)

def set_kth_bit(n, k):  # 1-indexed
    mask = 1 << (k - 1)
    m = n | mask
    print("value is", m, "and binary is", bin(m))

def unset_kth_bit(n, k):  # 1-indexed
    mask = 1 << (k - 1)
    m = n & ~(mask)
    print("value is", m, "and binary is", bin(m))

def toggle_kth_bit(n, k):  # 1-indexed
    mask = 1 << (k - 1)
    m = n ^ mask
    print("value is", m, "and binary is", bin(m))

def is_2_power(n):
    m = n & (n - 1)
    print(n, "is", ("not" if m != 0 else ""), "a power of 2")

# Usage:
# n = int(input())
# bin(n)[2:]  # Convert to binary string
# n.bit_count()  # Count number of 1s
# n.bit_length()  # Number of bits needed`}
            />
          </Section>

          {/* Difference Arrays */}
          <Section id="difference-arrays" title="Difference Arrays & Prefix/Suffix Sums">
            <div className="mb-4 p-4 bg-green-50 rounded-lg">
              <p className="text-gray-700">
                Efficient range update queries using difference arrays and prefix/suffix sum calculations.
              </p>
            </div>
            <CodeBlock
              language="python"
              code={`def range_update_queries(n, k):
    """Range updates using difference array"""
    diff = [0] * (n + 2)
    for i in range(k):
        l, r = map(int, input().split())
        diff[l] += 1
        diff[r + 1] -= 1
    
    # Convert to actual array
    for i in range(2, n + 1):
        diff[i] += diff[i - 1]
    
    return diff[1:n+1]

def prefix_sum(lst):
    """Calculate prefix sums"""
    pre = [0]
    for i in range(len(lst)):
        pre.append(pre[-1] + lst[i])
    return pre

def suffix_sum(lst):
    """Calculate suffix sums"""
    post = [0] * (len(lst) + 1)
    for i in range(len(lst) - 1, -1, -1):
        post[i] = post[i + 1] + lst[i]
    return post`}
            />
          </Section>

          {/* Fenwick Tree */}
          <Section id="fenwick-tree" title="Fenwick Tree (Binary Indexed Tree)">
            <div className="mb-4 p-4 bg-purple-50 rounded-lg">
              <p className="text-gray-700">
                Three variants: Point Update + Range Query, Range Update + Point Query, and Range Update + Range Query.
              </p>
            </div>
            
            <SubSection title="Range Update + Point Query">
              <CodeBlock
                language="python"
                code={`def sum(idx, F):
    running_sum = 0
    while idx > 0:
        running_sum += F[idx]
        right_most_set_bit = (idx & -idx)
        idx -= right_most_set_bit
    return running_sum

def add(idx, X, F):
    while idx < len(F):
        F[idx] += X
        right_most_set_bit = (idx & -idx)
        idx += right_most_set_bit

def point_query(idx, F):
    return sum(idx, F)

def range_update(l, r, X, F):
    add(l, X, F)
    add(r + 1, -X, F)

# Usage:
# n = 5
# arr = [-1e9, 1, 2, 3, 4, 5]  # 1-based indexing
# F = [0] * (n + 1)
# 
# # Build tree
# for i in range(1, n + 1):
#     range_update(i, i, arr[i], F)
# 
# # Query and update
# print(point_query(2, F))
# range_update(2, 4, 7, F)
# print(point_query(2, F))`}
              />
            </SubSection>

            <SubSection title="Range Update + Range Query">
              <CodeBlock
                language="python"
                code={`def sum(idx, F):
    running_sum = 0
    while idx > 0:
        running_sum += F[idx]
        right_most_set_bit = (idx & -idx)
        idx -= right_most_set_bit
    return running_sum

def add(idx, X, F):
    while idx < len(F):
        F[idx] += X
        right_most_set_bit = (idx & -idx)
        idx += right_most_set_bit

def pref_sum(idx, F1, F2):
    return sum(idx, F1) * idx - sum(idx, F2)

def range_query(L, R, F1, F2):
    return pref_sum(R, F1, F2) - pref_sum(L - 1, F1, F2)

def range_update(l, r, X, F1, F2):
    add(l, X, F1)
    add(r + 1, -X, F1)
    add(l, X * (l - 1), F2)
    add(r + 1, -(X * r), F2)

# Usage:
# n = 5
# arr = [-10**9, 1, 2, 3, 4, 5]
# F1 = [0] * (n + 1)
# F2 = [0] * (n + 1)
# 
# for i in range(1, n + 1):
#     range_update(i, i, arr[i], F1, F2)
# 
# print(range_query(2, 4, F1, F2))
# range_update(2, 4, 7, F1, F2)
# print(range_query(2, 4, F1, F2))`}
              />
            </SubSection>
          </Section>

          {/* Graphs */}
          <Section id="graphs" title="Graph Algorithms">
            <div className="mb-4 p-4 bg-indigo-50 rounded-lg">
              <p className="text-gray-700">
                Comprehensive collection of graph algorithms including traversals, shortest paths, MST, and more.
              </p>
            </div>

            <SubSection title="Graph Initialization">
              <CodeBlock
                language="python"
                code={`from collections import defaultdict, deque

def initialize_graph(undirected=False):
    """Create adjacency list representation"""
    graph = defaultdict(list)
    m = int(input())  # number of edges
    
    for _ in range(m):
        a, b = map(int, input().split())
        graph[a].append(b)
        if undirected:
            graph[b].append(a)
    
    return graph`}
              />
            </SubSection>

            <SubSection title="DFS & BFS">
              <CodeBlock
                language="python"
                code={`def dfs_iterative(graph, v, visited, parent):
    """Iterative DFS"""
    stack = [v]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for node in graph[v]:
            parent[node] = v
            stack.append(node)

def bfs_iterative(graph, v, visited, parent, dist):
    """Iterative BFS with distances"""
    parent[v] = -1
    dist[v] = 0
    q = deque([(v, 0)])
    visited.add(v)
    
    while q:
        v, dst = q.popleft()
        for node in graph[v]:
            if node not in visited:
                visited.add(node)
                q.append((node, dst + 1))
                parent[node] = v
                dist[node] = dst + 1`}
              />
            </SubSection>

            <SubSection title="Dijkstra's Algorithm">
              <CodeBlock
                language="python"
                code={`import heapq

def dijkstra(graph, n, src):
    """
    Dijkstra's shortest path algorithm
    graph: adjacency list with (neighbor, weight) tuples
    n: number of vertices (0-indexed)
    src: source vertex
    Returns: list of distances from source
    """
    dist = [float('inf')] * n
    dist[src] = 0
    pq = [(0, src)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if current_dist > dist[u]:
            continue
        
        for v, weight in graph[u]:
            distance = current_dist + weight
            if distance < dist[v]:
                dist[v] = distance
                heapq.heappush(pq, (distance, v))
    
    return dist`}
              />
            </SubSection>

            <SubSection title="Bellman-Ford Algorithm">
              <CodeBlock
                language="python"
                code={`def bellman_ford(graph, n, edges, src):
    """
    Bellman-Ford algorithm (handles negative weights)
    Returns: (has_negative_cycle, distances)
    """
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[src] = 0
    
    # Relax edges n-1 times
    for i in range(1, n):
        for a, b, w in edges:
            if dist[a] != INF:
                dist[b] = min(dist[b], dist[a] + w)
    
    # Check for negative cycles
    has_negative_cycle = False
    for a, b, w in edges:
        if dist[a] != INF and dist[a] + w < dist[b]:
            has_negative_cycle = True
            break
    
    return has_negative_cycle, dist`}
              />
            </SubSection>

            <SubSection title="Kruskal's MST with DSU">
              <CodeBlock
                language="python"
                code={`class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, x, y):
        s1 = self.find(x)
        s2 = self.find(y)
        if s1 != s2:
            if self.rank[s1] < self.rank[s2]:
                self.parent[s1] = s2
            elif self.rank[s1] > self.rank[s2]:
                self.parent[s2] = s1
            else:
                self.parent[s2] = s1
                self.rank[s1] += 1

def kruskals_mst(V, edges):
    """
    Kruskal's algorithm for MST
    edges: list of [u, v, weight]
    Returns: total cost of MST
    """
    edges.sort(key=lambda x: x[2])
    dsu = DSU(V)
    cost = 0
    count = 0
    
    for x, y, w in edges:
        if dsu.find(x) != dsu.find(y):
            dsu.union(x, y)
            cost += w
            count += 1
            if count == V - 1:
                break
    
    return cost`}
              />
            </SubSection>

            <SubSection title="Topological Sort">
              <CodeBlock
                language="python"
                code={`from collections import deque

def topological_sort(V, edges):
    """
    Topological sort using Kahn's algorithm
    Returns empty list if cycle detected
    """
    # Build adjacency list
    adj = [[] for _ in range(V)]
    indegree = [0] * V
    
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1
    
    # Queue with nodes having 0 indegree
    q = deque([i for i in range(V) if indegree[i] == 0])
    result = []
    
    while q:
        node = q.popleft()
        result.append(node)
        
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    
    # Check for cycle
    if len(result) != V:
        print("Graph contains cycle!")
        return []
    
    return result`}
              />
            </SubSection>

            <SubSection title="Tree Diameter">
              <CodeBlock
                language="python"
                code={`from collections import deque, defaultdict

def bfs_for_diameter(graph, start, visited):
    """BFS to find farthest node and distance"""
    q = deque([(start, 0)])
    levels = defaultdict(list)
    max_level = 0
    
    while q:
        node, level = q.popleft()
        if node in visited:
            continue
        visited.add(node)
        levels[level].append(node)
        max_level = max(max_level, level)
        
        for neighbor in graph[node]:
            q.append((neighbor, level + 1))
    
    return levels[max_level][0], max_level

def tree_diameter(graph):
    """Find diameter of tree using two BFS"""
    visited = set()
    farthest, _ = bfs_for_diameter(graph, 1, visited)
    
    visited = set()
    _, diameter = bfs_for_diameter(graph, farthest, visited)
    
    return diameter`}
              />
            </SubSection>

            <SubSection title="Flood Fill">
              <CodeBlock
                language="python"
                code={`from collections import deque

def floodfill(grid, start_r, start_c, target_color):
    """
    BFS-based flood fill
    grid: 2D list
    Returns: number of cells filled
    """
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    q = deque([(start_r, start_c)])
    count = 0
    
    while q:
        r, c = q.popleft()
        
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] != target_color or visited[r][c]):
            continue
        
        visited[r][c] = True
        count += 1
        
        # Add neighbors
        q.append((r + 1, c))
        q.append((r - 1, c))
        q.append((r, c + 1))
        q.append((r, c - 1))
    
    return count`}
              />
            </SubSection>
          </Section>

          {/* Modular Arithmetic */}
          <Section id="mod-ncr" title="Modular Arithmetic & Combinatorics">
            <div className="mb-4 p-4 bg-yellow-50 rounded-lg">
              <p className="text-gray-700 mb-2">
                <strong>Key Formulas:</strong>
              </p>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                <li>(a + b) % m = (a % m + b % m) % m</li>
                <li>(a - b) % m = (a % m - b % m) % m</li>
                <li>(a * b) % m = (a % m * b % m) % m</li>
                <li>a / b % m = a * b⁻¹ % m = a * b^(m-2) % m (if m is prime)</li>
              </ul>
            </div>
            <CodeBlock
              language="python"
              code={`def solve_binomial_coefficients():
    """
    Calculate nCr modulo prime using Fermat's Little Theorem
    Time: O(MAX + q)
    """
    MOD = 10**9 + 7
    MAX = 10**6
    
    # Precompute factorials
    fact = [1]
    for i in range(1, MAX + 1):
        fact.append((fact[-1] * i) % MOD)
    
    def mod_inverse(x):
        """Modular inverse using Fermat's Little Theorem"""
        return pow(x, MOD - 2, MOD)
    
    q = int(input())
    for _ in range(q):
        n, r = map(int, input().split())
        
        # nCr = n! / (r! * (n-r)!)
        numerator = fact[n]
        denominator = (fact[r] * fact[n - r]) % MOD
        result = (numerator * mod_inverse(denominator)) % MOD
        
        print(result)

# Additional modular arithmetic functions
def mod_exp(base, exp, mod):
    """Fast modular exponentiation"""
    return pow(base, exp, mod)

def extended_gcd(a, b):
    """Extended Euclidean Algorithm"""
    if b == 0:
        return a, 1, 0
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd, x, y`}
            />
          </Section>

          {/* Prime Numbers */}
          <Section id="primes" title="Prime Numbers">
            <div className="mb-4 p-4 bg-red-50 rounded-lg">
              <p className="text-gray-700">
                Sieve of Eratosthenes for efficient prime number generation up to a limit.
              </p>
            </div>
            <CodeBlock
              language="python"
              code={`def sieve_of_eratosthenes(limit):
    """
    Generate all primes up to limit
    Time: O(n log log n)
    Space: O(n)
    
    Returns:
        primes: list of prime numbers
        is_prime: boolean array for primality check
    """
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    p = 2
    while p * p <= limit:
        if is_prime[p]:
            # Mark all multiples as composite
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
        p += 1
    
    # Collect all primes
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    
    return primes, is_prime

# Usage:
# primes, is_prime = sieve_of_eratosthenes(1000000)
# 
# # Check if a number is prime
# if is_prime[17]:
#     print("17 is prime")
# 
# # Get first 10 primes
# print(primes[:10])

def is_prime_trial_division(n):
    """Check if n is prime using trial division"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    
    return True`}
            />
          </Section>

          {/* Priority Queue */}
          <Section id="priority-queue" title="Priority Queue (Heap)">
            <div className="mb-4 p-4 bg-pink-50 rounded-lg">
              <p className="text-gray-700">
                Python's heapq module for min-heap operations. For max-heap, negate priorities.
              </p>
            </div>
            <CodeBlock
              language="python"
              code={`import heapq

def create_priority_queue():
    """Create a new empty priority queue"""
    return []

def insert_with_priority(heap, priority, item):
    """Insert an item with a priority"""
    heapq.heappush(heap, (priority, item))
    return heap

def extract_min_with_priority(heap):
    """Remove and return the item with minimum priority"""
    if not heap:
        return None, None
    priority, item = heapq.heappop(heap)
    return priority, item

def is_empty(heap):
    """Check if the priority queue is empty"""
    return len(heap) == 0

def size(heap):
    """Return the number of elements in the priority queue"""
    return len(heap)

# For MAX heap, negate priorities:
# heapq.heappush(heap, (-priority, item))
# priority, item = heapq.heappop(heap)
# priority = -priority  # Convert back to positive

# Usage example:
# pq = create_priority_queue()
# insert_with_priority(pq, 5, "task1")
# insert_with_priority(pq, 1, "urgent")
# insert_with_priority(pq, 10, "low_priority")
# 
# while not is_empty(pq):
#     priority, task = extract_min_with_priority(pq)
#     print(f"Processing: {task} (priority: {priority})")`}
            />
          </Section>

          {/* SCC */}
          <Section id="scc" title="Strongly Connected Components (Tarjan's Algorithm)">
            <div className="mb-4 p-4 bg-indigo-50 rounded-lg">
              <p className="text-gray-700">
                Find all strongly connected components in a directed graph using Tarjan's algorithm in O(V+E) time.
              </p>
            </div>
            <CodeBlock
              language="python"
              code={`from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)
        self.Time = 0
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
    
    def scc_util(self, u, low, disc, stack_member, st):
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1
        stack_member[u] = True
        st.append(u)
        
        for v in self.graph[u]:
            if disc[v] == -1:
                self.scc_util(v, low, disc, stack_member, st)
                low[u] = min(low[u], low[v])
            elif stack_member[v]:
                low[u] = min(low[u], disc[v])
        
        w = -1
        if low[u] == disc[u]:
            component = []
            while w != u:
                w = st.pop()
                component.append(w)
                stack_member[w] = False
            print("SCC:", component)
    
    def find_sccs(self):
        disc = [-1] * self.V
        low = [-1] * self.V
        stack_member = [False] * self.V
        st = []
        
        for i in range(self.V):
            if disc[i] == -1:
                self.scc_util(i, low, disc, stack_member, st)

# Usage:
# g = Graph(5)
# g.add_edge(1, 0)
# g.add_edge(0, 2)
# g.add_edge(2, 1)
# g.add_edge(0, 3)
# g.add_edge(3, 4)
# g.find_sccs()`}
              />
          </Section>

          {/* Segment Tree */}
          <Section id="segment-tree" title="Segment Tree">
            <div className="mb-4 p-4 bg-teal-50 rounded-lg">
              <p className="text-gray-700">
                Efficient data structure for range queries and point updates. Supports sum, min, max, XOR, etc.
              </p>
            </div>
            <CodeBlock
              language="python"
              code={`def merge(l, r):
    """Merge operation - change for different queries"""
    return l + r
    # Other options: min(l, r), max(l, r), l ^ r, etc.

def build(arr, N):
    """Build segment tree from array"""
    tree = [0] * (2 * N)
    
    # Insert leaf nodes
    for i in range(N):
        tree[N + i] = arr[i]
    
    # Build tree by calculating parents
    for i in range(N - 1, 0, -1):
        tree[i] = merge(tree[i << 1], tree[i << 1 | 1])
    
    return tree

def update_tree_node(p, value, tree, n):
    """Update value at position p"""
    tree[p + n] = value
    p = p + n
    
    # Move upward and update parents
    i = p
    while i > 1:
        tree[i >> 1] = merge(tree[i], tree[i ^ 1])
        i >>= 1

def query(l, r, tree, n):
    """Query range [l, r) - r is exclusive"""
    res = 0
    l += n
    r += n
    
    while l < r:
        if l & 1:
            res = merge(res, tree[l])
            l += 1
        
        if r & 1:
            r -= 1
            res = merge(res, tree[r])
        
        l >>= 1
        r >>= 1
    
    return res

# Usage:
# a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# n = len(a)
# tree = build(a, n)
# 
# # Query sum in range [1, 3) (indices 1 and 2)
# print(query(1, 3, tree, n))
# 
# # Update element at index 2 to value 1
# update_tree_node(2, 1, tree, n)
# 
# # Query again
# print(query(1, 3, tree, n))

# Note: 
# - This uses 0-based indexing
# - Range queries have r as exclusive
# - For coordinate compression, use sorted list with binary search`}
              />
          </Section>

          {/* CPIT Tools */}
          <Section id="cpit-tools" title="CPIT - Competitive Programming Tools">
            <div className="mb-4 p-4 bg-orange-50 rounded-lg">
              <p className="text-gray-700">
                Command-line tools for parsing problems, testing code, checking ratings, and memory estimation.
              </p>
            </div>

            <SubSection title="Main Script (cpit.py)">
              <CodeBlock
                language="python"
                code={`import sys
from parser import parse_problem, parse_contest
from checker import test_code
from rating import get_difficulty
from mb import get_mem

if __name__ == "__main__":
    if sys.argv[1] == "parse":
        parse_type = sys.argv[2].lower()
        if parse_type in ['o', 'p', '1']:
            parse_problem(sys.argv[3])
        else:
            parse_contest(sys.argv[3])
    elif sys.argv[1] == "checker":
        executable = sys.argv[2]
        test_code(executable)
    elif sys.argv[1] == "rating":
        get_difficulty()
    elif sys.argv[1] == "mem":
        get_mem()

# Setup:
# alias cpit='python3 /path/to/cpit.py'
#
# Commands:
# cpit parse p <problem_link>  - Parse single problem
# cpit parse c <contest_id>    - Parse entire contest
# cpit checker <executable>    - Test code against samples
# cpit rating                  - Get problem rating
# cpit mem                     - Calculate array memory`}
              />
            </SubSection>

            <SubSection title="Checker for C++ (checker.py)">
              <div className="mb-2 text-sm text-gray-600">
                Automatically tests your compiled code against all .in files and compares with .out files.
              </div>
              <CodeBlock
                language="python"
                code={`# Usage: python checker.py <executable_name>
# Or with alias: cpit checker <executable_name>

import subprocess
import sys
import os
import re

class bcolors:
    OKGREEN = '\\033[92m'
    WARNING = '\\033[33m'
    FAIL = '\\033[91m'
    ENDC = '\\033[0m'
    BOLD = '\\033[1m'
    OKBLUE = '\\033[94m'

def get_tests():
    inputs = sorted(f for f in os.listdir('.') if f.endswith('.in'))
    outputs = sorted(f for f in os.listdir('.') if f.endswith('.out'))
    return inputs, outputs

def run_test(input_file, output_files, executable):
    results_lines = []
    expected_lines = []
    
    try:
        command = f'{executable} < {input_file}'
        p = subprocess.Popen(command, shell=True,
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, text=True)
        
        for line in p.stdout:
            line_strip = line.strip()
            if line_strip:
                results_lines.append(line_strip)
        
        p.wait()
    except Exception as e:
        print(f"{bcolors.FAIL}Error: {e}{bcolors.ENDC}")
        return [], []
    
    expected_file = input_file.replace(".in", ".out")
    if expected_file in output_files:
        with open(expected_file) as f:
            for line in f:
                line_strip = line.strip()
                if line_strip:
                    expected_lines.append(line_strip)
    
    return results_lines, expected_lines

def test_code(executable):
    input_files, output_files = get_tests()
    overall_success = True
    
    for input_file in input_files:
        results, expected = run_test(input_file, output_files, executable)
        print(f"Test {bcolors.BOLD}{input_file}{bcolors.ENDC}: ", end="")
        
        if len(results) != len(expected):
            print(f"{bcolors.FAIL}FAILED{bcolors.ENDC}")
            overall_success = False
        else:
            match = all(r.lower() == e.lower() 
                       for r, e in zip(results, expected))
            if match:
                print(f"{bcolors.OKGREEN}PASSED{bcolors.ENDC}")
            else:
                print(f"{bcolors.FAIL}FAILED{bcolors.ENDC}")
                overall_success = False
    
    if overall_success:
        print(f"\\n{bcolors.OKGREEN}ALL TESTS PASSED{bcolors.ENDC}")
    else:
        print(f"\\n{bcolors.FAIL}SOME TESTS FAILED{bcolors.ENDC}")`}
              />
            </SubSection>

            <SubSection title="Checker for Python (checker_py.py)">
              <div className="mb-2 text-sm text-gray-600">
                Tests Python scripts directly without compilation.
              </div>
              <CodeBlock
                language="python"
                code={`# Usage: python checker_py.py <python_file.py>

import subprocess
import sys
import os

def run_test(input_file, output_files, python_file):
    results_lines = []
    expected_lines = []
    
    try:
        with open(input_file) as inp:
            p = subprocess.Popen(
                [sys.executable, python_file],
                stdin=inp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = p.communicate()
            
            for line in stdout.strip().splitlines():
                line_strip = line.strip()
                if line_strip:
                    results_lines.append(line_strip)
    
    except Exception as e:
        print(f"Error: {e}")
        return [], []
    
    expected_file = input_file.replace(".in", ".out")
    if expected_file in output_files:
        with open(expected_file) as f:
            for line in f:
                line_strip = line.strip()
                if line_strip:
                    expected_lines.append(line_strip)
    
    return results_lines, expected_lines

# Same testing logic as checker.py`}
              />
            </SubSection>

            <SubSection title="Problem Parser (parser.py)">
              <div className="mb-2 text-sm text-gray-600">
                Scrapes Codeforces problems and contests to create local test files.
              </div>
              <CodeBlock
                language="python"
                code={`import cloudscraper
import re
import subprocess
import os

def parse_problem(LINK, path="./"):
    """Parse single Codeforces problem"""
    scraper = cloudscraper.create_scraper()
    response = scraper.get(LINK)
    
    # Extract input/output samples from HTML
    all_starts = [m.start() for m in re.finditer("<pre>", response.text)]
    all_ends = [m.start() for m in re.finditer("</pre>", response.text)]
    
    inputs = []
    outputs = []
    for i in range(len(all_starts)):
        if i & 1:
            outputs.append((all_starts[i], all_ends[i]))
        else:
            inputs.append((all_starts[i], all_ends[i]))
    
    # Create .in and .out files
    for i, (item, output_item) in enumerate(zip(inputs, outputs), 1):
        # Process input
        raw_str = response.text[item[0]:item[1]]
        raw_str = raw_str.replace("<br />", "\\n").replace("<pre>", "")
        raw_str = re.sub(r"<div class=.*?>", "", raw_str)
        raw_str = re.sub(r"</div>", "\\n", raw_str)
        
        with open(f"{path}{i}.in", "w") as f:
            f.write(raw_str.strip())
        
        # Process output
        raw_str = response.text[output_item[0]:output_item[1]]
        raw_str = raw_str.replace("<br />", "\\n").replace("<pre>", "")
        raw_str = re.sub(r"<div class=.*?>", "", raw_str)
        raw_str = re.sub(r"</div>", "\\n", raw_str)
        
        with open(f"{path}{i}.out", "w") as f:
            f.write(raw_str.strip())
    
    print(f"Parsed {len(inputs)} test cases")

def parse_contest(contest_id):
    """Parse entire Codeforces contest"""
    LINK = f"https://codeforces.com/contest/{contest_id}"
    scraper = cloudscraper.create_scraper()
    response = scraper.get(LINK)
    
    # Extract problem letters
    search_link = f"/contest/{contest_id}/problem/"
    all_starts = [m.start() for m in re.finditer(search_link, response.text)]
    
    problems = []
    for item in all_starts:
        cur_prob = ""
        cur_add = 0
        while response.text[item + len(search_link) + cur_add] != '"':
            cur_prob += response.text[item + len(search_link) + cur_add]
            cur_add += 1
        if not problems or problems[-1] != cur_prob:
            problems.append(cur_prob)
    
    # Create directories and parse each problem
    for problem in problems:
        os.makedirs(problem, exist_ok=True)
        problem_link = f"https://codeforces.com/contest/{contest_id}/problem/{problem}"
        parse_problem(problem_link, f"{problem}/")
    
    print(f"Parsed {len(problems)} problems: {', '.join(problems)}")`}
              />
            </SubSection>

            <SubSection title="Rating Checker (rating.py)">
              <div className="mb-2 text-sm text-gray-600">
                Extracts and displays the difficulty rating of a Codeforces problem.
              </div>
              <CodeBlock
                language="python"
                code={`import cloudscraper

def get_difficulty():
    """Get difficulty rating of a Codeforces problem"""
    try:
        link = input("Problem link: ")
        scraper = cloudscraper.create_scraper()
        response = scraper.get(link)
        
        # Find rating in HTML
        ind = response.text.find('title="Difficulty')
        while response.text[ind] != '*':
            ind += 1
        ind += 1
        
        rating = ""
        while response.text[ind] != '\\r':
            rating += response.text[ind]
            ind += 1
        
        rating = int(rating)
        
        # Color codes by difficulty
        difficulties = [1199, 1399, 1599, 1899, 2099, 2399, 10000]
        colors = ["", "\\u001b[38;5;10m", "\\u001b[38;5;14m",
                 "\\u001b[38;5;25m", "\\u001b[38;5;99m", 
                 "\\u001b[38;5;3m", "\\u001b[38;5;9m"]
        
        for i in range(len(difficulties)):
            if rating < difficulties[i]:
                print(f"Rating: \\033[1m{colors[i]}{rating}\\033[0m")
                break
    
    except Exception:
        print("No rating found")`}
              />
            </SubSection>

            <SubSection title="Memory Calculator (mb.py)">
              <div className="mb-2 text-sm text-gray-600">
                Estimates memory usage for arrays in competitive programming.
              </div>
              <CodeBlock
                language="python"
                code={`def get_mem():
    """Calculate memory usage of an array"""
    array_len = input("Length of the array: ")
    
    # Handle scientific notation
    if 'e' in array_len:
        begin_num = float(array_len[:array_len.index('e')])
        end_num = int(array_len[array_len.index('e') + 1:])
        array_len = begin_num * (10 ** end_num)
    else:
        array_len = float(array_len)
    
    data_type = input("int, ll, double? ").strip().lower()
    
    print(f"Array Length = {array_len}")
    
    if data_type == "int":
        # 4 bytes per int
        print(f"Memory: {array_len * 4 / 1e6:.2f} MB")
    else:
        # 8 bytes per ll/double
        print(f"Memory: {array_len * 8 / 1e6:.2f} MB")

# Common memory limits:
# Codeforces: usually 256 MB
# AtCoder: usually 1024 MB
# 
# Rule of thumb:
# - int[10^6] ≈ 4 MB
# - long long[10^6] ≈ 8 MB
# - int[10^8] ≈ 400 MB (might TLE/MLE)

if __name__ == "__main__":
    get_mem()`}
              />
            </SubSection>

            <SubSection title="Setup Instructions">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-800 mb-2">Installation</h4>
                <CodeBlock
                  language="bash"
                  code={`# Clone the repository
git clone https://github.com/C1XTEEN/CPIT.git
cd CPIT

# Install dependencies
pip install -r requirements.txt

# Add alias to ~/.bashrc or ~/.zshrc
echo "alias cpit='python3 $(pwd)/cpit.py'" >> ~/.bashrc
source ~/.bashrc`}
                />
                <h4 className="font-semibold text-gray-800 mt-4 mb-2">Usage Examples</h4>
                <CodeBlock
                  language="bash"
                  code={`# Parse a single problem
cpit parse p https://codeforces.com/contest/1234/problem/A

# Parse entire contest
cpit parse c 1234

# Test your code (after compiling)
g++ -o solution solution.cpp
cpit checker ./solution

# Test Python code
cpit checker "python3 solution.py"

# Get problem rating
cpit rating

# Calculate memory
cpit mem`}
                />
              </div>
            </SubSection>
          </Section>

          {/* Footer */}
          <footer className="mt-16 pt-8 border-t border-gray-300 text-center text-gray-600">
            <p className="mb-2">
              <strong>CP Template Repository</strong> - Your one-stop resource for competitive programming
            </p>
            <p className="text-sm">
              All algorithms tested and optimized for competitive programming contests
            </p>
            <p className="text-xs mt-4 text-gray-500">
              Pro tip: Use Ctrl/Cmd + F to quickly search for specific algorithms or techniques
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
}