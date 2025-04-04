from py2neo import Graph
import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
import numpy as np
import pandas as pd
from torch_geometric.nn import GraphSAGE, GATConv
from torch_geometric.data import Data
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ----- Tiền xử lý văn bản: Biên dịch sẵn các pattern regex -----
exp_skills_pattern = re.compile(r'(\d+ years of experience in [\w\s]+)')
expert_exp_pattern = re.compile(r'(Expert in [\w\s]+|Experienced in [\w\s]+)')
worked_projects_pattern = re.compile(r'(worked on [\w\s]+)')


def extract_entities(about_text):
  skills = exp_skills_pattern.findall(about_text)
  expertise = expert_exp_pattern.findall(about_text)
  projects = worked_projects_pattern.findall(about_text)
  return {
    "skills": skills,
    "projects": projects,
    "expertise": expertise
  }


# ----- Kết nối đến Neo4j -----
NEO4J_URI = "neo4j+s://fa2fd127.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "k6y0bLBbHmLw5g-lopuQFKvIsEvjyTig7Y2r-p7aPOc"

try:
  graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
  print("✅ Kết nối Neo4j thành công!")
except Exception as e:
  print(f"❌ Lỗi kết nối Neo4j: {e}")
  exit()

query = """
MATCH (e:Employee)-[:HAS_ABOUT]->(a:About)
RETURN id(e) AS source, id(a) AS target, e.name AS employee_name, a.type AS about_text
"""
result = graph.run(query).to_data_frame()

if result.empty:
  print("⚠ Không tìm thấy dữ liệu quan hệ Employee-About")
  exit()
else:
  print(result)

# ----- Chuyển dữ liệu sang tensor và tạo đồ thị -----
edge_index = torch.tensor(result[['source', 'target']].values,
                          dtype=torch.long).t().contiguous()
n_nodes = int(edge_index.max().item()) + 1
x = torch.randn(n_nodes, 16)  # Random feature cho mỗi node
data = Data(x=x, edge_index=edge_index)

# Tạo từ điển ánh xạ ID -> Tên (Employee/About)
id_to_name = {row['source']: row['employee_name'] for _, row in
              result.iterrows()}
id_to_name.update(
    {row['target']: row['about_text'] for _, row in result.iterrows()})

# ----- Negative Sampling được cải tiến -----
positive_edges = edge_index.t().tolist()
positive_set = set(tuple(edge) for edge in positive_edges)
all_nodes = np.arange(n_nodes)
negative_edges = []
while len(negative_edges) < len(positive_edges):
  src = int(np.random.choice(all_nodes))
  tgt = int(np.random.choice(all_nodes))
  if (src, tgt) not in positive_set:
    negative_edges.append([src, tgt])

# Kết hợp các cạnh dương và âm
combined_edges = positive_edges + negative_edges
y = torch.cat(
    [torch.ones(len(positive_edges)), torch.zeros(len(negative_edges))])
edge_index_combined = torch.tensor(combined_edges,
                                   dtype=torch.long).t().contiguous()
data.edge_index = edge_index_combined

# ----- Tách tập dữ liệu thành train/validation cho việc tune ngưỡng -----
indices = np.arange(y.size(0))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42,
                                      stratify=y.numpy())
train_mask = torch.tensor([i in train_idx for i in range(y.size(0))])
val_mask = torch.tensor([i in val_idx for i in range(y.size(0))])


# ----- Định nghĩa mô hình: Cho phép lựa chọn giữa GraphSAGE và GAT, áp dụng temperature scaling -----
class LinkPredictor(nn.Module):
  def __init__(self, in_channels, hidden_channels, use_attention=True,
      temperature=0.7):
    super().__init__()
    self.use_attention = use_attention
    self.temperature = temperature
    if use_attention:
      # Sử dụng GAT với 2 heads (attention)
      self.conv = GATConv(in_channels, hidden_channels, heads=2, concat=False)
    else:
      self.sage = GraphSAGE(in_channels, hidden_channels, num_layers=100)
    self.mlp = nn.Sequential(
        nn.Linear(hidden_channels * 2, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

  def forward(self, data):
    if self.use_attention:
      x = self.conv(data.x, data.edge_index)
    else:
      x = self.sage(data.x, data.edge_index)
    edge_embeds = torch.cat([x[data.edge_index[0]], x[data.edge_index[1]]],
                            dim=1)
    logits = self.mlp(edge_embeds).squeeze()
    return logits / self.temperature  # Hiệu chuẩn với temperature scaling


# Khởi tạo mô hình (có thể chuyển sang use_attention=True nếu muốn dùng attention)
model = LinkPredictor(in_channels=32, hidden_channels=64, use_attention=True,
                      temperature=0.7)
optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# ----- Huấn luyện mô hình với Early Stopping dựa trên tập validation -----
best_val_loss = float('inf')
patience = 10
trigger_times = 0
num_epochs = 10

for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()
  output = model(data)
  loss = criterion(output[train_mask], y[train_mask])
  loss.backward()
  optimizer.step()

  model.eval()
  with torch.no_grad():
    val_output = model(data)
    val_loss = criterion(val_output[val_mask], y[val_mask])
  if epoch % 10 == 0:
    print(
      f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
  if val_loss.item() < best_val_loss:
    best_val_loss = val_loss.item()
    trigger_times = 0
    best_model_state = model.state_dict()
  else:
    trigger_times += 1
    if trigger_times >= patience:
      print("Early stopping triggered")
      break

model.load_state_dict(best_model_state)


# ----- Hàm tune_threshold: Tìm ngưỡng tối ưu dựa trên F1-score trên tập validation -----
def tune_threshold(model, data, y, mask):
  model.eval()
  with torch.no_grad():
    outputs = model(data)
    probs = torch.sigmoid(outputs[mask]).cpu().numpy()
    labels = y[mask].cpu().numpy()
  thresholds = np.linspace(0, 1, 101)
  best_thresh = 0.5
  best_f1 = 0.0
  for t in thresholds:
    preds = (probs >= t).astype(int)
    f1 = f1_score(labels, preds)
    if f1 > best_f1:
      best_f1 = f1
      best_thresh = t
  print(f"Tối ưu ngưỡng: {best_thresh:.2f} với F1: {best_f1:.4f}")
  return best_thresh


optimal_threshold = tune_threshold(model, data, y, val_mask)

# ----- Dự đoán quan hệ ẩn -----
model.eval()
with torch.no_grad():
  all_outputs = model(data)
  all_probs = torch.sigmoid(all_outputs).cpu().numpy()
# Sử dụng ngưỡng tối ưu để chọn các cạnh dự đoán
predicted_edges = edge_index_combined[:,
                  all_probs >= optimal_threshold].t().tolist()

print("\n🔍 Quan hệ ẩn mới được phát hiện:")
if not predicted_edges:
  print("⚠ Không tìm thấy quan hệ ẩn nào")
else:
  for idx, (src, tgt) in enumerate(predicted_edges):
    src_name = id_to_name.get(src, f"Node {src}")
    tgt_name = id_to_name.get(tgt, f"Node {tgt}")
    # Lấy điểm dự đoán cho cạnh (có thể dựa vào vị trí tương ứng)
    pred_score = all_probs[(all_outputs >= optimal_threshold).nonzero()[idx]]
    if tgt_name is None or tgt_name.strip() == "":
      print(f"⚠ {src_name} -> Không có thông tin từ About (BỎ QUA)")
      continue
    entities = extract_entities(tgt_name)
    if entities["skills"] or entities["projects"] or entities["expertise"]:
      for skill in entities["skills"]:
        confidence = 0.5 * pred_score + 0.3 * min(1.0, len(
          skill.split()) / 10) + 0.2 * 0.5
        if confidence > 0.6:
          print(
            f"🛠️ {src_name} -> {skill} (HAS_DETAIL_EXPERIENCE) [Confidence: {confidence:.2f}]")
      for project in entities["projects"]:
        confidence = 0.5 * pred_score + 0.3 * min(1.0, len(
          project.split()) / 10) + 0.2 * 0.5
        if confidence > 0.6:
          print(
            f"🛠️ {src_name} -> {project} (WORKED_ON) [Confidence: {confidence:.2f}]")
      for exp in entities["expertise"]:
        confidence = 0.5 * pred_score + 0.3 * min(1.0, len(
          exp.split()) / 10) + 0.2 * 0.5
        if confidence > 0.6:
          print(
            f"🛠️ {src_name} -> {exp} (EXPERTISE) [Confidence: {confidence:.2f}]")
    else:
      print(f"⚠ {src_name} -> Không tìm thấy thực thể nào (BỎ QUA)")

# ----- Cập nhật quan hệ vào Neo4j -----
if predicted_edges:
  print("\n📡 Cập nhật vào Neo4j...")
  for idx, (src, tgt) in enumerate(predicted_edges):
    src_name = id_to_name.get(src, f"Node {src}")
    tgt_name = id_to_name.get(tgt, f"Node {tgt}")
    with torch.no_grad():
      pred_score = all_probs[(all_outputs >= optimal_threshold).nonzero()[idx]]
    if tgt_name is None or tgt_name.strip() == "":
      continue
    entities = extract_entities(tgt_name)
    relationships = []
    if entities["skills"]:
      for skill in entities["skills"]:
        confidence = 0.5 * pred_score + 0.3 * min(1.0, len(
          skill.split()) / 10) + 0.2 * 0.5
        if confidence > 0.6:
          relationships.append(
              (src_name, "HAS_DETAIL_EXPERIENCE", skill, confidence))
    if entities["projects"]:
      for project in entities["projects"]:
        confidence = 0.5 * pred_score + 0.3 * min(1.0, len(
          project.split()) / 10) + 0.2 * 0.5
        if confidence > 0.6:
          relationships.append((src_name, "WORKED_ON", project, confidence))
    if entities["expertise"]:
      for exp in entities["expertise"]:
        confidence = 0.5 * pred_score + 0.3 * min(1.0, len(
          exp.split()) / 10) + 0.2 * 0.5
        if confidence > 0.6:
          relationships.append((src_name, "EXPERTISE", exp, confidence))
    for (source, relation, target, conf) in relationships:
      cypher_query = f"""
              MERGE (e:Employee {{name: '{source}'}})
              MERGE (t:Detail_Exp {{name: '{target}'}})
              MERGE (e)-[r:{relation}]->(t)
              SET r.confidence = {conf}
              """
      graph.run(cypher_query)
      print(
        f"✅ {source} -[:{relation} (conf:{conf:.2f})]-> {target} đã được thêm vào Neo4j")
  print("🎯 Cập nhật hoàn tất!")
else:
  print("⚠ Không có quan hệ mới để cập nhật.")
