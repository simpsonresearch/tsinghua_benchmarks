# Welcome!

I wanted to see how the new Tsinghua shortest path algorithm performs against Dijkstra's & Bellman-Ford algorithms.

## Running the App

To run the app (python3 required),

```bash
cd src
python3 main.py
```

## Performance Analysis

<img width="1000" height="661" alt="Screenshot 2025-08-11 at 9 38 23 AM" src="https://github.com/user-attachments/assets/bc1ea62f-0a6d-4431-8e37-e78eae5ec730" />

```bash
--- Performance Analysis ---
This will test algorithms on graphs of increasing size.

Testing with 50 nodes...
  dijkstra: 0.36 ms
  bellman_ford: 0.26 ms
  tsinghua: 0.32 ms

Testing with 100 nodes...
  dijkstra: 0.87 ms
  bellman_ford: 0.89 ms
  tsinghua: 0.84 ms

Testing with 200 nodes...
  dijkstra: 7.79 ms
  bellman_ford: 11.26 ms
  tsinghua: 7.95 ms

Testing with 500 nodes...
  dijkstra: 10.82 ms
  bellman_ford: 15.30 ms
  tsinghua: 13.66 ms
```
