import os
import re

celeb_df_dir = r"D:\Download\SatyaDrishti_Datasets\Celeb DF"
synthesis_dir = os.path.join(celeb_df_dir, "Celeb-synthesis")

if os.path.exists(synthesis_dir):
    edges = set()
    all_nodes = set()
    for f in os.listdir(synthesis_dir):
        if f.lower().endswith(".mp4"):
            ids = [int(x[2:]) for x in re.findall(r'(id\d+)', f)]
            if len(ids) == 2:
                u, v = min(ids), max(ids)
                edges.add((u, v))
                all_nodes.add(u)
                all_nodes.add(v)
                
    # Also read real folders to find all node IDs
    for folder in ["Celeb-real", "YouTube-real"]:
        folder_path = os.path.join(celeb_df_dir, folder)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                ids = [int(x[2:]) for x in re.findall(r'(id\d+)', f)]
                for u in ids:
                    all_nodes.add(u)
                    
    # Find connected components
    adj = {n: set() for n in all_nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    visited = set()
    components = []
    for n in all_nodes:
        if n not in visited:
            comp = set()
            queue = [n]
            visited.add(n)
            while queue:
                curr = queue.pop(0)
                comp.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)
            
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Total components: {len(components)}")
    for idx, comp in enumerate(components):
        print(f"Component {idx}: size {len(comp)}, nodes {sorted(list(comp))}")
else:
    print("Celeb-synthesis directory not found.")
