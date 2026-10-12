import requests, csv, time
Q = ('("GUI agent" | "GUI agents" | "web agent" | "web agents" | "computer-use agent" | "computer use agent" '
     '| "desktop agent" | "browser agent" | "UI agent" | "GUI automation" | "UI automation" | "web navigation" '
     '| "robotic process automation" | WebArena | OSWorld | Mind2Web | MiniWoB | AndroidWorld) '
     '+ (symbolic | "neuro-symbolic" | neurosymbolic | PDDL | "classical planning" | "automated planning" '
     '| "hierarchical task network" | "formal verification" | "model checking" | "temporal logic" '
     '| "runtime verification" | verifier | "cognitive architecture" | "state machine" | "transition graph")')
url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
params = {"query": Q, "year": "2020-",
          "fields": "title,abstract,year,venue,authors,externalIds,citationCount,publicationTypes"}
rows, token = [], None
while True:
    if token: params["token"] = token
    r = requests.get(url, params=params, timeout=60); r.raise_for_status(); d = r.json()
    rows += d["data"]; token = d.get("token")
    print(len(rows), "of", d.get("total"))
    if not token: break
    time.sleep(3.5)
with open("s2_optionB_strict.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["Authors", "Title", "Year", "Source title", "DOI", "Abstract", "Link"])
    for p in rows:
        ids = p.get("externalIds") or {}
        link = f"https://arxiv.org/abs/{ids['ArXiv']}" if ids.get("ArXiv") else ""
        w.writerow(["; ".join(a["name"] for a in p.get("authors") or []), p["title"], p.get("year"),
                    p.get("venue") or ("arXiv" if ids.get("ArXiv") else ""), ids.get("DOI", ""),
                    p.get("abstract") or "", link])