import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
from py2neo import Graph
from neo4j import GraphDatabase
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data

# Nhập cấu hình và cấu trúc dữ liệu từ các tệp riêng biệt
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    MODEL_PARAMS, TRAINING_PARAMS,
    CONFIDENCE_THRESHOLDS, CONFIDENCE_WEIGHTS
)
from entity_patterns import ENTITY_PATTERNS
from context_keywords import CONTEXT_KEYWORDS

# ============================
# Kết nối Neo4j
# ============================
def connect_to_neo4j():
    """Thiết lập kết nối đến cơ sở dữ liệu Neo4j."""
    try:
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("✅ Kết nối Neo4j thành công!")
        return graph
    except Exception as e:
        print(f"❌ Lỗi kết nối Neo4j: {e}")
        exit()

# ============================
# Trích xuất dữ liệu
# ============================
def extract_data_from_neo4j(graph):
    """Trích xuất dữ liệu node và mối quan hệ từ Neo4j."""
    # Query để trích xuất các mối quan hệ hiện có
    query = """
    MATCH (e:Employee)-[r]->(n)
    WHERE type(r) <> 'HAS_ABOUT' AND any(label IN labels(n) WHERE label IN ['Nationality', 'Language', 'Project', 'Technology', 'Seniority'])
    RETURN id(e) AS source, id(n) AS target, e.name AS employee_name, labels(n)[0] AS node_type, n.name AS node_name, type(r) AS relationship_type
    UNION
    MATCH (e:Employee)-[:HAS_ABOUT]->(a:About)
    RETURN id(e) AS source, id(a) AS target, e.name AS employee_name, 'About' AS node_type, a.type AS node_name, 'HAS_ABOUT' AS relationship_type
    """
    result = graph.run(query).to_data_frame()

    if result.empty:
        print("⚠ Không tìm thấy mối quan hệ Employee-About nào")
        exit()
    else:
        print(f"Tìm thấy {len(result)} mối quan hệ")

    return result

def extract_existing_technology_relationships(graph):
    """Truy vấn riêng các mối quan hệ giữa nhân viên và công nghệ (Technology)."""
    query = """
    MATCH (e:Employee)-[r]->(t:Technology)
    RETURN e.name AS employee_name, t.name AS technology_name, type(r) AS relationship_type
    """
    existing_tech_relationships = graph.run(query).to_data_frame()

    # Tạo từ điển để theo dõi các mối quan hệ công nghệ hiện có cho mỗi nhân viên
    employee_tech_relationships = {}
    if not existing_tech_relationships.empty:
        for _, row in existing_tech_relationships.iterrows():
            employee = row['employee_name']
            tech = row['technology_name']
            if employee not in employee_tech_relationships:
                employee_tech_relationships[employee] = set()
            employee_tech_relationships[employee].add(tech.lower())

        print(f"Tìm thấy mối quan hệ công nghệ hiện có cho {len(employee_tech_relationships)} nhân viên")
    else:
        print("Không tìm thấy mối quan hệ công nghệ nào")

    return employee_tech_relationships

# ============================
# Xây dựng đồ thị
# ============================
def prepare_graph_data(result):

    # Chuyển đổi thành tensor
    edge_index = torch.tensor(result[['source', 'target']].values,
                              dtype=torch.long).t().contiguous()
    n_nodes = max(edge_index.flatten()).item() + 1
    x = torch.randn(n_nodes, 64)  # Vector đặc trưng ngẫu nhiên cho mỗi node
    data = Data(x=x, edge_index=edge_index)

    # Tạo từ điển id_to_info để ánh xạ ID node trong đồ thị sang thông tin thực thể
    # Lưu thông tin tên và loại cho mỗi node
    id_to_info = {}
    for _, row in result.iterrows():
        id_to_info[row['source']] = {'name': row['employee_name'], 'type': 'Employee'}
        id_to_info[row['target']] = {'name': row['node_name'],'type': row['node_type']}

    # Tạo các mẫu âm (negative samples)
    positive_edges = edge_index.t().tolist()
    all_nodes = set(range(n_nodes))
    negative_edges = []

    while len(negative_edges) < len(positive_edges):
        src, tgt = random.sample(sorted(all_nodes), 2)
        if [src, tgt] not in positive_edges:
            negative_edges.append([src, tgt])

    y = torch.cat(
        [torch.ones(len(positive_edges)), torch.zeros(len(negative_edges))])
    train_edge_index = torch.tensor(positive_edges + negative_edges,
                                    dtype=torch.long).t().contiguous()

    return data, train_edge_index, y, id_to_info

# ============================
# Định nghĩa mô hình
# ============================
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.2):
        """
        Tham số:
            in_channels: Số lượng đặc trưng đầu vào
            hidden_channels: Số lượng đặc trưng ẩn
            num_layers: Số lớp GraphSAGE
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        self.sage = GraphSAGE(in_channels, hidden_channels, num_layers=num_layers, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        """Tạo biểu diễn node."""
        x = self.sage(x, edge_index)
        return x

    def predict_links(self, x, edge_index_to_predict):
        """Dự đoán khả năng tồn tại liên kết giữa các cặp node."""
        edge_embeds = torch.cat(
            [x[edge_index_to_predict[0]], x[edge_index_to_predict[1]]], dim=1)
        return self.mlp(edge_embeds).squeeze()

# ============================
# Huấn luyện mô hình
# ============================
def train_model(model, data, train_edge_index, train_y, epochs=100, lr=0.01, print_interval=10):
    """Huấn luyện mô hình LinkPredictor."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print("\n🔄 Đang huấn luyện mô hình GraphSAGE...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        node_embeddings = model(data.x, data.edge_index)
        output = model.predict_links(node_embeddings, train_edge_index)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        if epoch % print_interval == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model

# ============================
# Trích xuất thực thể
# ============================
def extract_entities(about_text, entity_types=None):
    """
    Trích xuất thực thể từ văn bản About dựa trên các loại thực thể được chỉ định.

    Tham số:
        about_text: Văn bản để trích xuất thực thể
        entity_types: Danh sách các loại thực thể cần trích xuất

    Trả về:
        Từ điển các thực thể đã trích xuất theo loại
    """
    if entity_types is None:
        entity_types = ENTITY_PATTERNS.keys()

    entities = {entity_type: [] for entity_type in entity_types}

    # Xử lý từng câu riêng biệt để có ngữ cảnh tốt hơn
    sentences = re.split(r'[.;]', about_text)  # Tách bằng dấu chấm và chấm phẩy
    sentences = [s.strip() for s in sentences if s.strip()]

    # Áp dụng mẫu cho từng câu và loại thực thể được yêu cầu
    for sentence in sentences:
        for entity_type in entity_types:
            if entity_type in ENTITY_PATTERNS:
                for pattern in ENTITY_PATTERNS[entity_type]:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    if matches:
                        # Trích xuất từng thực thể từ danh sách được phân tách bởi dấu phẩy
                        for match in matches:
                            if isinstance(match, tuple):  # Đối với các nhóm regex
                                match = match[0]  # Lấy nhóm đầu tiên

                            # Xử lý danh sách thực thể được phân tách bằng dấu phẩy
                            if ',' in match:
                                # Nếu là danh sách phân tách bằng dấu phẩy, tách và xử lý từng mục
                                split_entities = [m.strip() for m in re.split(r',\s*', match)]
                                for split_entity in split_entities:
                                    # Kiểm tra lại từng thực thể được tách với mẫu
                                    if re.search(pattern, split_entity, re.IGNORECASE):
                                        entities[entity_type].append(split_entity)
                            else:
                                entities[entity_type].append(match.strip())

    # Xử lý đặc biệt cho Experience_Level để trích xuất thời gian và công nghệ liên quan
    if 'Experience_Level' in entity_types and entities['Experience_Level']:
        refined_experiences = []
        for exp in entities['Experience_Level']:
            # Trích xuất số năm
            duration_pattern = r'(\d+\+?)\s*years?'
            duration_match = re.search(duration_pattern, exp, re.IGNORECASE)
            if duration_match:
                duration = duration_match.group(1)

                # Cố gắng trích xuất công nghệ/lĩnh vực mà kinh nghiệm liên quan đến
                tech_pattern = r'experience\s+(?:in|with|of)\s+([\w\s,\.\-]+)'
                tech_match = re.search(tech_pattern, exp, re.IGNORECASE)

                # Nếu không khớp với mẫu tiêu chuẩn, thử mẫu thay thế
                if not tech_match:
                    tech_pattern = r'experience\s+(?:in|with|of)\s+the\s+([\w\s,\.\-]+)'
                    tech_match = re.search(tech_pattern, exp, re.IGNORECASE)

                if tech_match:
                    tech = tech_match.group(1).strip()
                    # Làm sạch chuỗi công nghệ
                    tech = re.sub(r'[,\.]$', '', tech).strip()
                    refined_experiences.append(f"{duration} years in {tech}")
                else:
                    refined_experiences.append(f"{duration} years")

        entities['Experience_Level'] = refined_experiences

    # Làm sạch thực thể (loại bỏ trùng lặp, cắt ngắn các mục dài)
    for entity_type in entity_types:
        entities[entity_type] = list(set(entities[entity_type]))
        # Cắt ngắn các thực thể dài
        entities[entity_type] = [entity[:100] if len(entity) > 100 else entity
                               for entity in entities[entity_type]]

    return entities

def calculate_confidence(prediction_score, entity_text, about_text, entity_type):
    """
    Tính toán điểm tin cậy cho một thực thể được trích xuất dựa trên nhiều yếu tố.

    Tham số:
        prediction_score: Điểm từ mô hình GraphSAGE
        entity_text: Văn bản thực thể được trích xuất
        about_text: Văn bản gốc mà thực thể được trích xuất từ đó
        entity_type: Loại của thực thể

    Trả về:
        Tuple của (điểm_tin_cậy, từ_điển_chi_tiết)
    """
    # Lấy trọng số dựa trên loại thực thể
    weights = CONFIDENCE_WEIGHTS.get(entity_type, CONFIDENCE_WEIGHTS['default'])
    alpha = weights['prediction']  # Trọng số cho dự đoán GraphSAGE
    beta = weights['specificity']  # Trọng số cho tính cụ thể của thực thể
    gamma = weights['context']     # Trọng số cho chất lượng ngữ cảnh

    # Đo lường tính cụ thể của thực thể (dài hơn và chi tiết hơn thì tốt hơn)
    specificity_score = min(1.0, len(entity_text.split()) / 10)

    # Điểm ngữ cảnh mặc định
    context_score = 0.0

    # Tìm vị trí thực thể trong văn bản about
    entity_pos = about_text.lower().find(entity_text.lower())
    if entity_pos >= 0:
        # Lấy ngữ cảnh xung quanh thực thể
        start = max(0, entity_pos - 30)
        end = min(len(about_text), entity_pos + len(entity_text) + 30)
        context = about_text[start:end]

        # Lấy từ khóa thích hợp cho loại thực thể này
        keywords = CONTEXT_KEYWORDS.get(entity_type,
                                       ["experience", "skill", "worked", "knowledge", "years"])

        # Đếm số lần xuất hiện từ khóa trong ngữ cảnh
        keyword_count = sum(
            1 for keyword in keywords if keyword.lower() in context.lower())
        context_score = min(1.0, keyword_count / 3)

    # Tính toán tổng độ tin cậy
    confidence = alpha * prediction_score + beta * specificity_score + gamma * context_score

    return confidence, {
        "prediction": prediction_score,
        "specificity": specificity_score,
        "context": context_score
    }

def determine_relationship_type(source_type, target_type):
    """
    Xác định loại mối quan hệ dựa trên loại node nguồn và đích.

    Tham số:
        source_type: Loại của node nguồn
        target_type: Loại của node đích

    Trả về:
        Chuỗi đại diện cho loại mối quan hệ
    """
    if source_type == 'Employee':
        if target_type == 'Skill':
            return 'HAS_DETAIL_SKILL'
        elif target_type == 'Technology':
            return 'HAS_TECHNOLOGY'
        elif target_type == 'Experience_Level':
            return 'HAS_EXPERIENCE_LEVEL'
        elif target_type == 'Project':
            return 'WORKED_ON'
        elif target_type == 'Industry':
            return 'HAS_INDUSTRY_EXPERIENCE'
        elif target_type == 'Soft_Skill':
            return 'HAS_SOFT_SKILL'

    # Trường hợp mặc định
    return 'RELATED_TO'

def create_temp_entity_embedding(entity_text, entity_type, node_embeddings, id_to_info):
    """
    Tạo một biểu diễn tạm thời cho một thực thể mới dựa trên các thực thể tương tự trong đồ thị.

    Tham số:
        entity_text: Văn bản của thực thể mới
        entity_type: Loại của thực thể (Technology, Skill, v.v.)
        node_embeddings: Biểu diễn node hiện tại từ mô hình
        id_to_info: Từ điển ánh xạ ID node sang thông tin node

    Trả về:
        Một tensor biểu diễn cho thực thể mới
    """
    # Tìm các node hiện có cùng loại để sử dụng làm tham chiếu
    similar_nodes = []

    # Lấy ID của các node hiện có cùng loại
    for node_id, info in id_to_info.items():
        if info['type'] == entity_type:
            # Tính toán độ tương đồng văn bản (đơn giản là sự trùng lặp từ)
            entity_words = set(entity_text.lower().split())
            node_words = set(info['name'].lower().split())

            # Tính độ tương đồng
            if len(entity_words) > 0 and len(node_words) > 0:
                intersection = len(entity_words.intersection(node_words))
                union = len(entity_words.union(node_words))
                similarity = intersection / union if union > 0 else 0

                similar_nodes.append((node_id, similarity))

    # Sắp xếp theo độ tương đồng (cao nhất trước)
    similar_nodes.sort(key=lambda x: x[1], reverse=True)

    # Nếu có node tương tự, sử dụng trung bình có trọng số của top 3 làm biểu diễn
    if similar_nodes:
        # Lấy top 3 hoặc nhiều nhất có thể
        top_nodes = similar_nodes[:min(3, len(similar_nodes))]

        # Nếu không có node tương tự nào có độ tương đồng > 0, sử dụng biểu diễn ngẫu nhiên
        if sum(sim for _, sim in top_nodes) == 0:
            return torch.randn(node_embeddings.shape[1])

        # Tính trung bình có trọng số biểu diễn
        weighted_sum = torch.zeros(node_embeddings.shape[1])
        total_weight = 0

        for node_id, similarity in top_nodes:
            if similarity > 0:
                weighted_sum += node_embeddings[node_id] * similarity
                total_weight += similarity

        if total_weight > 0:
            return weighted_sum / total_weight

    # Phương án dự phòng: Trả về một biểu diễn ngẫu nhiên thiên về trung bình của tất cả các thực thể cùng loại
    type_embeddings = [node_embeddings[node_id] for node_id, info in
                      id_to_info.items()
                      if info['type'] == entity_type]

    if type_embeddings:
        # Trả về biểu diễn ngẫu nhiên thiên về trung bình của loại này
        avg_embedding = torch.stack(type_embeddings).mean(dim=0)
        random_factor = 0.3
        return avg_embedding * (1 - random_factor) + torch.randn_like(
            avg_embedding) * random_factor

    # Phương án cuối cùng: biểu diễn hoàn toàn ngẫu nhiên
    return torch.randn(node_embeddings.shape[1])

def get_existing_technology_node(graph, technology_name):
    """
    Kiểm tra xem một node công nghệ với tên đã cho có tồn tại trong cơ sở dữ liệu hay không.

    Tham số:
        graph: Kết nối đồ thị Neo4j
        technology_name: Tên của công nghệ cần kiểm tra

    Trả về:
        Boolean cho biết node có tồn tại hay không
    """
    query = """
    MATCH (t:Technology {name: $tech_name})
    RETURN count(t) > 0 AS exists
    """
    result = graph.evaluate(query, tech_name=technology_name)
    return result

# ============================
# Hàm xử lý chính
# ============================
def process_about_texts(graph, result, node_embeddings, model, id_to_info, employee_tech_relationships):
    """
    Xử lý các node About và trích xuất thực thể.

    Tham số:
        graph: Kết nối đồ thị Neo4j
        result: DataFrame chứa dữ liệu đồ thị
        node_embeddings: Biểu diễn node từ mô hình đã huấn luyện
        model: Mô hình LinkPredictor đã huấn luyện
        id_to_info: Từ điển ánh xạ ID node sang thông tin node
        employee_tech_relationships: Từ điển theo dõi các mối quan hệ công nghệ hiện có

    Trả về:
        Thống kê về các thực thể và mối quan hệ đã trích xuất
    """
    print("\n🔍 Đang xử lý các node About và trích xuất thực thể...")
    employees_with_about = result[result['node_type'] == 'About']

    # Bộ đếm
    total_hidden_nodes_found = 0
    total_hidden_relationships_found = 0
    hidden_nodes_by_type = {
        'Technology': 0,
        'Skill': 0,
        'Soft_Skill': 0,
        'Experience_Level': 0,
        'Project': 0
    }
    hidden_relationships_by_type = {
        'HAS_TECHNOLOGY': 0,
        'HAS_DETAIL_SKILL': 0,
        'HAS_SOFT_SKILL': 0,
        'HAS_EXPERIENCE_LEVEL': 0,
        'WORKED_ON': 0
    }

    for _, row in employees_with_about.iterrows():
        employee_id = row['source']
        about_id = row['target']
        employee_name = row['employee_name']
        about_text = row['node_name']

        # Bỏ qua nếu văn bản About trống
        if about_text is None or about_text.strip() == "":
            print(f"⚠ {employee_name} có văn bản About trống (BỎ QUA)")
            continue

        print(f"\n📑 Đang xử lý văn bản About cho {employee_name}:")

        # Trích xuất thực thể từ văn bản About
        entity_types = ['Skill', 'Technology', 'Experience_Level', 'Project',
                        'Soft_Skill']
        extracted_entities = extract_entities(about_text, entity_types)

        # Xử lý từng loại thực thể
        new_relationships = []

        for entity_type, entities in extracted_entities.items():
            if not entities:
                continue

            print(f"  Tìm thấy {len(entities)} thực thể {entity_type}")

            for entity in entities:
                # Xử lý đặc biệt cho thực thể Technology - KIỂM TRA MỐI QUAN HỆ HIỆN CÓ
                if entity_type == 'Technology':
                    # Chuyển đổi thành chữ thường để so sánh không phân biệt chữ hoa/thường
                    entity_lower = entity.lower()

                    # Kiểm tra xem nhân viên đã có mối quan hệ với công nghệ này chưa (bất kể loại mối quan hệ)
                    if employee_name in employee_tech_relationships and entity_lower in employee_tech_relationships[
                        employee_name]:
                        print(f"  ⏩ Bỏ qua {entity}: {employee_name} đã có mối quan hệ với công nghệ này")
                        continue

                # Tạo biểu diễn tạm thời cho thực thể mới
                temp_entity_embedding = create_temp_entity_embedding(entity, entity_type,
                                                                     node_embeddings, id_to_info)

                # Lấy biểu diễn của nhân viên
                employee_embedding = node_embeddings[employee_id]

                # Tính điểm dự đoán sử dụng mô hình đã huấn luyện
                # Nối biểu diễn của nhân viên và thực thể
                edge_embedding = torch.cat([employee_embedding, temp_entity_embedding],
                                           dim=0).unsqueeze(0)

                # Sử dụng mô hình để dự đoán xác suất liên kết
                with torch.no_grad():
                    prediction_score_tensor = torch.sigmoid(
                        model.mlp(edge_embedding).squeeze())
                    prediction_score = prediction_score_tensor.item()

                # Tính toán độ tin cậy tổng thể
                confidence, details = calculate_confidence(
                    prediction_score, entity, about_text, entity_type
                )

                threshold = CONFIDENCE_THRESHOLDS.get(entity_type, 0.6)
                # Chỉ giữ lại các mối quan hệ có độ tin cậy cao hơn ngưỡng
                if confidence > threshold:
                    relationship_type = determine_relationship_type('Employee', entity_type)
                    new_relationships.append({
                        'employee_name': employee_name,
                        'entity': entity,
                        'entity_type': entity_type,
                        'relationship_type': relationship_type,
                        'confidence': confidence,
                        'prediction_score': prediction_score,
                        'context_score': details['context'],
                        'specificity_score': details['specificity']
                    })

                    print(
                        f"  🔄 {entity_type}: {entity[:50]}{'...' if len(entity) > 50 else ''} [Conf: {confidence:.2f}, Pred: {prediction_score:.2f}]")

        # Thêm mối quan hệ vào Neo4j
        for rel in new_relationships:
            # Đối với thực thể Technology, trước tiên kiểm tra xem node đã tồn tại chưa
            node_exists = False
            if rel['entity_type'] == 'Technology':
                node_exists = get_existing_technology_node(graph, rel['entity'])
            else:
                # Kiểm tra xem node đã tồn tại chưa đối với các loại thực thể khác
                check_node_query = f"""
          MATCH (n:{rel['entity_type']} {{name: $name}})
          RETURN count(n) > 0 AS exists
          """
                node_exists = graph.evaluate(check_node_query, name=rel['entity'])

            # Tạo truy vấn Cypher để thêm mối quan hệ
            # Sử dụng MERGE cho cả node và mối quan hệ để đảm bảo chúng tồn tại mà không bị trùng lặp
            cypher_query = f"""
                MATCH (e:Employee {{name: $employee_name}})
                MERGE (t:{rel['entity_type']} {{name: $entity_name}})
                MERGE (e)-[r:{rel['relationship_type']}]->(t)
                SET r.confidence = $confidence,
                    r.prediction_score = $prediction_score,
                    r.context_score = $context_score,
                    r.specificity_score = $specificity_score,
                    r.extracted_from = 'about_text'
                """

            # Thực thi truy vấn
            try:
                graph.run(
                    cypher_query,
                    employee_name=rel['employee_name'],
                    entity_name=rel['entity'],
                    confidence=rel['confidence'],
                    prediction_score=rel['prediction_score'],
                    context_score=rel['context_score'],
                    specificity_score=rel['specificity_score']
                )

                # Cập nhật bộ đếm thống kê
                total_hidden_relationships_found += 1
                hidden_relationships_by_type[rel['relationship_type']] = hidden_relationships_by_type.get(
                    rel['relationship_type'], 0) + 1

                if not node_exists:
                    total_hidden_nodes_found += 1
                    hidden_nodes_by_type[rel['entity_type']] = hidden_nodes_by_type.get(rel['entity_type'], 0) + 1

                # Thông báo khác nhau tùy thuộc vào việc node đã tồn tại trước đó hay chưa
                if node_exists:
                    print(
                        f"  ✅ Đã thêm mối quan hệ đến node hiện có: {rel['employee_name']} -[:{rel['relationship_type']} ({rel['confidence']:.2f})]-> {rel['entity_type']}:{rel['entity'][:30]}..."
                    )
                else:
                    print(
                        f"  ✅ Đã thêm node và mối quan hệ mới: {rel['employee_name']} -[:{rel['relationship_type']} ({rel['confidence']:.2f})]-> {rel['entity_type']}:{rel['entity'][:30]}..."
                    )

                # Nếu đây là thực thể Technology, cập nhật theo dõi trong bộ nhớ
                if rel['entity_type'] == 'Technology':
                    if rel['employee_name'] not in employee_tech_relationships:
                        employee_tech_relationships[rel['employee_name']] = set()
                    employee_tech_relationships[rel['employee_name']].add(rel['entity'].lower())

            except Exception as e:
                print(f"  ❌ Lỗi khi thêm mối quan hệ: {e}")

    # Trả về thống kê
    return {
        'total_hidden_nodes': total_hidden_nodes_found,
        'total_hidden_relationships': total_hidden_relationships_found,
        'hidden_nodes_by_type': hidden_nodes_by_type,
        'hidden_relationships_by_type': hidden_relationships_by_type
    }

def print_summary_statistics(stats):
    """In thống kê tóm tắt của quá trình trích xuất."""
    print("\n📊 THỐNG KÊ TÓM TẮT:")
    print(f"Tổng số node ẩn được phát hiện: {stats['total_hidden_nodes']}")
    print(f"Tổng số mối quan hệ ẩn được phát hiện: {stats['total_hidden_relationships']}")

    if stats['total_hidden_nodes'] > 0:
        print("\nNode ẩn theo loại:")
        for node_type, count in stats['hidden_nodes_by_type'].items():
            if count > 0:
                print(f"  - {node_type}: {count}")

    if stats['total_hidden_relationships'] > 0:
        print("\nMối quan hệ ẩn theo loại:")
        for rel_type, count in stats['hidden_relationships_by_type'].items():
            if count > 0:
                print(f"  - {rel_type}: {count}")


def main():
    # Kết nối đến Neo4j
    graph = connect_to_neo4j()

    # Trích xuất dữ liệu
    result = extract_data_from_neo4j(graph)
    employee_tech_relationships = extract_existing_technology_relationships(graph)

    # Chuẩn bị dữ liệu đồ thị
    data, train_edge_index, train_y, id_to_info = prepare_graph_data(result)

    # Khởi tạo và huấn luyện mô hình
    model = LinkPredictor(
        in_channels=MODEL_PARAMS['in_channels'],
        hidden_channels=MODEL_PARAMS['hidden_channels'],
        num_layers=MODEL_PARAMS['num_layers'],
        dropout=MODEL_PARAMS['dropout']
    )

    model = train_model(
        model, data, train_edge_index, train_y,
        epochs=TRAINING_PARAMS['epochs'],
        lr=TRAINING_PARAMS['learning_rate'],
        print_interval=TRAINING_PARAMS['print_interval']
    )

    # Lấy biểu diễn node từ mô hình đã huấn luyện
    with torch.no_grad():
        node_embeddings = model(data.x, data.edge_index)

    # Xử lý văn bản About
    stats = process_about_texts(
        graph, result, node_embeddings, model,
        id_to_info, employee_tech_relationships
    )


    print_summary_statistics(stats)

    print("\n🎯 Xử lý hoàn tất!")

if __name__ == "__main__":
    main()