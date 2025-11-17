import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
from pyvis.network import Network
import tempfile
import os
from pathlib import Path
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 페이지 설정
st.set_page_config(
    page_title="Network Analysis Viewer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .node-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_network_data(prefix='autophagy'):
    """네트워크 데이터 로드
    
    Args:
        prefix: 파일명 접두사 ('autophagy' 또는 'research_autophagy')
    """
    try:
        # 폴더 경로 설정
        if prefix == 'research_autophagy':
            folder = 'Research'
            # 연구용: research_autophagy_protein_ppi_network_edgelist.csv
            ppi_file = os.path.join(folder, f'{prefix}_protein_ppi_network_edgelist.csv')
        else:
            folder = 'All'
            # 전체용: autophagy_ppi_network_edgelist.csv
            ppi_file = os.path.join(folder, f'{prefix}_ppi_network_edgelist.csv')
        
        ppi_df = pd.read_csv(ppi_file, header=None, names=['target1', 'target2', 'score'])
        G_ppi = nx.from_pandas_edgelist(ppi_df, 'target1', 'target2', edge_attr='score', create_using=nx.Graph())
        
        # GGI 네트워크
        ggi_file = os.path.join(folder, f'{prefix}_gene_network_edgelist.csv')
        ggi_df = pd.read_csv(ggi_file, header=None, names=['target1', 'target2', 'score'])
        G_ggi = nx.from_pandas_edgelist(ggi_df, 'target1', 'target2', edge_attr='score', create_using=nx.Graph())
        
        return {
            'PPI': {'graph': G_ppi, 'df': ppi_df},
            'GGI': {'graph': G_ggi, 'df': ggi_df}
        }
    except FileNotFoundError as e:
        st.error(f"파일을 찾을 수 없습니다: {e.filename}")
        return None
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

@st.cache_data
def load_centrality_data(network_name, prefix='autophagy'):
    """중심성 데이터 로드
    
    Args:
        network_name: 네트워크 이름 ('PPI' 또는 'GGI')
        prefix: 파일명 접두사 ('autophagy' 또는 'research_autophagy')
    """
    try:
        # 폴더 경로 설정
        if prefix == 'research_autophagy':
            folder = 'research'
        else:
            folder = 'all'
        
        centrality_file = os.path.join(folder, f'{prefix}_{network_name}_centrality_analysis.csv')
        if os.path.exists(centrality_file):
            return pd.read_csv(centrality_file, index_col=0)
        return None
    except Exception as e:
        st.warning(f"중심성 데이터를 로드할 수 없습니다: {e}")
        return None

def calculate_simple_centrality(graph):
    """간단한 중심성 계산 (캐시된 데이터가 없을 경우)"""
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, weight='score', normalized=True)
    
    # Closeness centrality (정규화 포함)
    closeness_raw = nx.closeness_centrality(graph, 
                                             distance=lambda u, v, d: 1.0 / max(d.get('score', 1.0), 1e-6))
    max_closeness = max(closeness_raw.values()) if closeness_raw.values() else 1.0
    closeness = {node: val / max_closeness if max_closeness > 0 else 0.0 
                 for node, val in closeness_raw.items()}
    
    eigenvector = nx.eigenvector_centrality(graph, weight='score', max_iter=1000, tol=1.0e-6)
    
    return pd.DataFrame({
        'Degree': pd.Series(degree),
        'Betweenness': pd.Series(betweenness),
        'Closeness': pd.Series(closeness),
        'Eigenvector': pd.Series(eigenvector)
    })

def create_interactive_network(graph, centrality_df, selected_nodes=None, max_nodes=200, network_type='PPI'):
    """인터랙티브 네트워크 그래프 생성
    
    Args:
        network_type: 네트워크 타입 ('PPI' 또는 'GGI') - 색상 팔레트 결정
    """
    # 노드 수가 많으면 서브그래프 생성
    if len(graph.nodes()) > max_nodes:
        if centrality_df is not None and not centrality_df.empty:
            top_nodes = centrality_df.nlargest(max_nodes, 'Betweenness').index.tolist()
            subgraph = graph.subgraph(top_nodes)
        else:
            # Degree 기준으로 상위 노드 선택
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_nodes = [node for node, _ in top_nodes]
            subgraph = graph.subgraph(top_nodes)
    else:
        subgraph = graph
        top_nodes = list(subgraph.nodes())
    
    # 선택된 노드가 있으면 해당 노드와 연결된 노드들도 포함
    if selected_nodes:
        extended_nodes = set(selected_nodes)
        for node in selected_nodes:
            if node in subgraph:
                extended_nodes.update(subgraph.neighbors(node))
        subgraph = graph.subgraph(list(extended_nodes) + top_nodes[:max_nodes//2])
    
    # 초기 위치 계산 (안정화를 위해 미리 계산)
    initial_pos = nx.spring_layout(subgraph, k=1.5, iterations=100, seed=42)
    
    # 엣지 색상 설정 (네트워크 타입에 따라)
    if network_type == 'PPI':
        edge_color = "rgba(34, 139, 34, 0.4)"  # 포레스트 그린
    else:
        edge_color = "rgba(255, 215, 0, 0.4)"  # 골드
    
    # Pyvis 네트워크 생성 (더 밝고 예쁜 배경)
    net = Network(height="600px", width="100%", bgcolor="#f8f9fa", font_color="#2c3e50")
    net.set_options(f"""
    {{
      "nodes": {{
        "borderWidth": 2,
        "borderColor": "#ffffff",
        "font": {{
          "size": 14,
          "face": "Arial",
          "color": "#2c3e50"
        }}
      }},
      "edges": {{
        "color": {{
          "color": "{edge_color}",
          "highlight": "rgba(255, 215, 0, 0.8)"
        }},
        "width": 1.5,
        "smooth": {{
          "type": "continuous"
        }}
      }},
      "physics": {{
        "enabled": true,
        "stabilization": {{
          "enabled": true,
          "iterations": 500,
          "fit": true,
          "onlyDynamicEdges": false
        }},
        "barnesHut": {{
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.3
        }},
        "solver": "barnesHut",
        "timestep": 0.35
      }},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false
      }}
    }}
    """)
    
    # 노드 추가
    centrality_dict = {}
    if centrality_df is not None and not centrality_df.empty:
        for node in subgraph.nodes():
            if node in centrality_df.index:
                centrality_dict[node] = {
                    'degree': centrality_df.loc[node, 'Degree'],
                    'betweenness': centrality_df.loc[node, 'Betweenness'],
                    'closeness': centrality_df.loc[node, 'Closeness'],
                    'eigenvector': centrality_df.loc[node, 'Eigenvector']
                }
    
    for node in subgraph.nodes():
        # 노드 크기 결정 (Degree centrality 기준)
        if node in centrality_dict:
            size = 20 + centrality_dict[node]['degree'] * 30
            title = f"Node: {node}<br>"
            title += f"Degree: {centrality_dict[node]['degree']:.4f}<br>"
            title += f"Betweenness: {centrality_dict[node]['betweenness']:.4f}<br>"
            title += f"Closeness: {centrality_dict[node]['closeness']:.4f}<br>"
            title += f"Eigenvector: {centrality_dict[node]['eigenvector']:.4f}"
        else:
            size = 15
            degree = subgraph.degree(node)
            title = f"Node: {node}<br>Degree: {degree}"
        
        # 선택된 노드는 다른 색상
        if selected_nodes and node in selected_nodes:
            color = "#FFD700"  # 금색
            border_width = 5
        else:
            # Betweenness centrality와 네트워크 타입에 따라 색상 결정
            if node in centrality_dict:
                betweenness = centrality_dict[node]['betweenness']
                # 네트워크 타입에 따른 색상 팔레트
                if network_type == 'PPI':
                    # PPI: 초록색 계열
                    if betweenness < 0.33:
                        t = betweenness / 0.33
                        r = int(0 + (34 - 0) * t)
                        g = int(100 + (139 - 100) * t)
                        b = int(0 + (34 - 0) * t)
                    elif betweenness < 0.66:
                        t = (betweenness - 0.33) / 0.33
                        r = int(34 + (50 - 34) * t)
                        g = int(139 + (205 - 139) * t)
                        b = int(34 + (50 - 34) * t)
                    else:
                        t = (betweenness - 0.66) / 0.34
                        r = int(50 + (124 - 50) * t)
                        g = int(205 + (252 - 205) * t)
                        b = int(50 + (0 - 50) * t)
                else:  # GGI
                    # GGI: 노란색 계열
                    if betweenness < 0.33:
                        t = betweenness / 0.33
                        r = int(255 + (255 - 255) * t)
                        g = int(165 + (215 - 165) * t)
                        b = int(0 + (0 - 0) * t)
                    elif betweenness < 0.66:
                        t = (betweenness - 0.33) / 0.33
                        r = int(255 + (255 - 255) * t)
                        g = int(215 + (255 - 215) * t)
                        b = int(0 + (0 - 0) * t)
                    else:
                        t = (betweenness - 0.66) / 0.34
                        r = int(255 + (255 - 255) * t)
                        g = int(255 + (255 - 255) * t)
                        b = int(0 + (224 - 0) * t)
                color = f"rgb({r}, {g}, {b})"
            else:
                # 기본 색상: 네트워크 타입에 따라
                if network_type == 'PPI':
                    color = "#228B22"  # 포레스트 그린
                else:
                    color = "#FFD700"  # 골드
            border_width = 2
        
        # 초기 위치 설정 (안정화를 위해)
        x, y = initial_pos[node]
        net.add_node(node, label=node, size=size, color=color, 
                    title=title, borderWidth=border_width, x=x*100, y=y*100)
    
    # 엣지 추가
    for u, v, data in subgraph.edges(data=True):
        score = data.get('score', 1.0)
        width = 1 + score * 2  # 엣지 두께
        net.add_edge(u, v, value=score, width=width, title=f"Score: {score:.4f}")
    
    # HTML 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as tmp_file:
        net.save_graph(tmp_file.name)
    
    # 파일을 읽은 후 닫기
    with open(tmp_file.name, 'r', encoding='utf-8') as f:
        html_string = f.read()
    
    # 파일이 닫힌 후 삭제
    os.unlink(tmp_file.name)
    
    # 노드 클릭 이벤트를 위한 JavaScript 추가
    # (참고: Streamlit에서는 직접적인 이벤트 전달이 제한적이므로,
    #  사용자는 그래프에서 노드를 더블클릭하거나 사이드바에서 선택해야 합니다)
    click_script = """
    <script>
    // 노드 더블클릭 시 URL 파라미터 업데이트 (선택사항)
    // 실제 구현은 Streamlit의 session state를 사용하는 것이 더 좋습니다
    </script>
    """
    html_string = html_string.replace('</body>', click_script + '</body>')
    
    return html_string, list(subgraph.nodes())

def create_3d_network(graph, centrality_df, selected_nodes=None, max_nodes=200, layout_method='spring', network_type='PPI'):
    """3D 네트워크 시각화 생성 (Plotly 사용)
    
    Args:
        network_type: 네트워크 타입 ('PPI' 또는 'GGI') - 색상 팔레트 결정
    """
    # 노드 수가 많으면 서브그래프 생성
    if len(graph.nodes()) > max_nodes:
        if centrality_df is not None and not centrality_df.empty:
            top_nodes = centrality_df.nlargest(max_nodes, 'Betweenness').index.tolist()
            subgraph = graph.subgraph(top_nodes)
        else:
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_nodes = [node for node, _ in top_nodes]
            subgraph = graph.subgraph(top_nodes)
    else:
        subgraph = graph
        top_nodes = list(subgraph.nodes())
    
    # 선택된 노드가 있으면 해당 노드와 연결된 노드들도 포함
    if selected_nodes:
        extended_nodes = set(selected_nodes)
        for node in selected_nodes:
            if node in subgraph:
                extended_nodes.update(subgraph.neighbors(node))
        subgraph = graph.subgraph(list(extended_nodes) + top_nodes[:max_nodes//2])
    
    # 3D 레이아웃 계산
    nodes_list = list(subgraph.nodes())
    n_nodes = len(nodes_list)
    
    if layout_method == 'spring':
        # Spring layout를 3D로 확장
        pos_2d = nx.spring_layout(subgraph, dim=2, k=1, iterations=50, seed=42)
        # Z축은 degree centrality로 설정
        if centrality_df is not None and not centrality_df.empty:
            z_pos = [centrality_df.loc[node, 'Degree'] if node in centrality_df.index else 0 
                    for node in nodes_list]
        else:
            degrees = dict(subgraph.degree())
            max_degree = max(degrees.values()) if degrees.values() else 1
            z_pos = [degrees.get(node, 0) / max_degree for node in nodes_list]
        
        x_pos = [pos_2d[node][0] for node in nodes_list]
        y_pos = [pos_2d[node][1] for node in nodes_list]
    elif layout_method == 'tsne':
        # t-SNE를 사용한 3D 레이아웃
        adjacency_matrix = nx.adjacency_matrix(subgraph, nodelist=nodes_list).todense()
        perplexity = min(30, max(5, n_nodes - 1))
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        pos_3d = tsne.fit_transform(adjacency_matrix)
        x_pos = pos_3d[:, 0].tolist()
        y_pos = pos_3d[:, 1].tolist()
        z_pos = pos_3d[:, 2].tolist()
    else:  # 'pca'
        # PCA를 사용한 3D 레이아웃
        adjacency_matrix = nx.adjacency_matrix(subgraph, nodelist=nodes_list).todense()
        pca = PCA(n_components=3, random_state=42)
        pos_3d = pca.fit_transform(adjacency_matrix)
        x_pos = pos_3d[:, 0].tolist()
        y_pos = pos_3d[:, 1].tolist()
        z_pos = pos_3d[:, 2].tolist()
    
    # 중심성 정보 준비
    centrality_dict = {}
    if centrality_df is not None and not centrality_df.empty:
        for node in nodes_list:
            if node in centrality_df.index:
                centrality_dict[node] = {
                    'degree': centrality_df.loc[node, 'Degree'],
                    'betweenness': centrality_df.loc[node, 'Betweenness'],
                    'closeness': centrality_df.loc[node, 'Closeness'],
                    'eigenvector': centrality_df.loc[node, 'Eigenvector']
                }
    
    # 노드 색상 및 크기 설정
    node_colors = []
    node_sizes = []
    node_texts = []
    
    # 네트워크 타입에 따른 색상 팔레트
    def get_color_from_betweenness(betweenness, network_type):
        """Betweenness centrality와 네트워크 타입에 따라 색상 반환"""
        if network_type == 'PPI':
            # PPI: 초록색 계열 그라데이션 (어두운 초록 -> 밝은 초록 -> 라임 그린)
            if betweenness < 0.33:
                t = betweenness / 0.33
                r = int(0 + (34 - 0) * t)
                g = int(100 + (139 - 100) * t)
                b = int(0 + (34 - 0) * t)
            elif betweenness < 0.66:
                t = (betweenness - 0.33) / 0.33
                r = int(34 + (50 - 34) * t)
                g = int(139 + (205 - 139) * t)
                b = int(34 + (50 - 34) * t)
            else:
                t = (betweenness - 0.66) / 0.34
                r = int(50 + (124 - 50) * t)
                g = int(205 + (252 - 205) * t)
                b = int(50 + (0 - 50) * t)
        else:  # GGI
            # GGI: 노란색 계열 그라데이션 (주황 노란색 -> 노란색 -> 밝은 노란색)
            if betweenness < 0.33:
                t = betweenness / 0.33
                r = int(255 + (255 - 255) * t)
                g = int(165 + (215 - 165) * t)
                b = int(0 + (0 - 0) * t)
            elif betweenness < 0.66:
                t = (betweenness - 0.33) / 0.33
                r = int(255 + (255 - 255) * t)
                g = int(215 + (255 - 215) * t)
                b = int(0 + (0 - 0) * t)
            else:
                t = (betweenness - 0.66) / 0.34
                r = int(255 + (255 - 255) * t)
                g = int(255 + (255 - 255) * t)
                b = int(0 + (224 - 0) * t)
        return f'rgb({r}, {g}, {b})'
    
    for i, node in enumerate(nodes_list):
        if selected_nodes and node in selected_nodes:
            # 선택된 노드는 금색 계열
            node_colors.append('rgb(255, 215, 0)')
            node_sizes.append(18)
        elif node in centrality_dict:
            betweenness = centrality_dict[node]['betweenness']
            node_colors.append(get_color_from_betweenness(betweenness, network_type))
            node_sizes.append(8 + centrality_dict[node]['degree'] * 15)
        else:
            # 기본 색상: 네트워크 타입에 따라
            if network_type == 'PPI':
                node_colors.append('rgb(34, 139, 34)')  # 포레스트 그린
            else:
                node_colors.append('rgb(255, 215, 0)')  # 골드
            node_sizes.append(8)
        
        # 툴팁 텍스트
        if node in centrality_dict:
            node_texts.append(
                f"{node}<br>"
                f"Degree: {centrality_dict[node]['degree']:.4f}<br>"
                f"Betweenness: {centrality_dict[node]['betweenness']:.4f}<br>"
                f"Closeness: {centrality_dict[node]['closeness']:.4f}<br>"
                f"Eigenvector: {centrality_dict[node]['eigenvector']:.4f}"
            )
        else:
            degree = subgraph.degree(node)
            node_texts.append(f"{node}<br>Degree: {degree}")
    
    # 엣지 좌표 생성
    edge_x = []
    edge_y = []
    edge_z = []
    edge_info = []
    
    node_to_index = {node: i for i, node in enumerate(nodes_list)}
    
    for u, v, data in subgraph.edges(data=True):
        u_idx = node_to_index[u]
        v_idx = node_to_index[v]
        
        edge_x.extend([x_pos[u_idx], x_pos[v_idx], None])
        edge_y.extend([y_pos[u_idx], y_pos[v_idx], None])
        edge_z.extend([z_pos[u_idx], z_pos[v_idx], None])
        
        score = data.get('score', 1.0)
        edge_info.append(f"{u} - {v}<br>Score: {score:.4f}")
    
    # Plotly 그래프 생성
    fig = go.Figure()
    
    # 엣지 추가 (네트워크 타입에 따른 색상)
    if network_type == 'PPI':
        edge_color = 'rgba(34, 139, 34, 0.3)'  # 포레스트 그린
    else:
        edge_color = 'rgba(255, 215, 0, 0.3)'  # 골드
    
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=0.8, color=edge_color),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # 노드 추가
    fig.add_trace(go.Scatter3d(
        x=x_pos, y=y_pos, z=z_pos,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.5, color='rgba(255, 255, 255, 0.8)'),
            opacity=0.9
        ),
        text=node_texts,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text="3D Network Visualization",
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(
                title="X", 
                showbackground=True, 
                backgroundcolor='rgba(240, 240, 240, 0.1)',
                showgrid=True, 
                gridcolor='rgba(200, 200, 200, 0.3)',
                zeroline=False
            ),
            yaxis=dict(
                title="Y", 
                showbackground=True, 
                backgroundcolor='rgba(240, 240, 240, 0.1)',
                showgrid=True, 
                gridcolor='rgba(200, 200, 200, 0.3)',
                zeroline=False
            ),
            zaxis=dict(
                title="Z", 
                showbackground=True, 
                backgroundcolor='rgba(240, 240, 240, 0.1)',
                showgrid=True, 
                gridcolor='rgba(200, 200, 200, 0.3)',
                zeroline=False
            ),
            bgcolor='rgba(250, 250, 250, 1)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(255, 255, 255, 1)',
        plot_bgcolor='rgba(250, 250, 250, 1)'
    )
    
    return fig, list(subgraph.nodes())

def get_node_info(graph, node, centrality_df):
    """노드 정보 가져오기"""
    if node not in graph:
        return None
    
    info = {
        'node': node,
        'degree': graph.degree(node),
        'neighbors': list(graph.neighbors(node)),
        'edges': []
    }
    
    # 연결된 엣지 정보
    for neighbor in graph.neighbors(node):
        edge_data = graph.get_edge_data(node, neighbor)
        score = edge_data.get('score', 0) if edge_data else 0
        info['edges'].append({
            'target': neighbor,
            'score': score
        })
    
    # 중심성 정보
    if centrality_df is not None and node in centrality_df.index:
        info['centrality'] = {
            'degree': centrality_df.loc[node, 'Degree'],
            'betweenness': centrality_df.loc[node, 'Betweenness'],
            'closeness': centrality_df.loc[node, 'Closeness'],
            'eigenvector': centrality_df.loc[node, 'Eigenvector']
        }
    
    return info

def render_network_tab(network_type, network_data, selected_node_key, prefix='autophagy'):
    """네트워크 탭 렌더링 함수
    
    Args:
        network_type: 네트워크 타입 ('PPI' 또는 'GGI')
        network_data: 네트워크 데이터 딕셔너리
        selected_node_key: 선택된 노드를 저장할 session state 키
        prefix: 파일명 접두사 ('autophagy' 또는 'research_autophagy')
    """
    selected_network = network_data[network_type]
    graph = selected_network['graph']
    
    # Session state 초기화
    if selected_node_key not in st.session_state:
        st.session_state[selected_node_key] = None
    
    # 중심성 데이터 로드
    centrality_df = load_centrality_data(network_type, prefix=prefix)
    if centrality_df is None or centrality_df.empty:
        with st.spinner("중심성을 계산하는 중..."):
            centrality_df = calculate_simple_centrality(graph)
    
    # 노드 검색
    st.sidebar.subheader(f"🔍 {network_type} 노드 검색")
    all_nodes = sorted(list(graph.nodes()))
    search_key = f"{network_type}_search"
    search_term = st.sidebar.text_input("노드 이름 검색", key=search_key, value="")
    
    if search_term:
        filtered_nodes = [node for node in all_nodes if search_term.lower() in node.lower()]
        if filtered_nodes:
            default_index = 0
            if st.session_state[selected_node_key] in filtered_nodes:
                default_index = filtered_nodes.index(st.session_state[selected_node_key])
            selected_node = st.sidebar.selectbox(
                "노드 선택", 
                options=filtered_nodes, 
                index=default_index,
                key=f"{network_type}_node_select"
            )
            st.session_state[selected_node_key] = selected_node
        else:
            st.sidebar.warning("검색 결과가 없습니다.")
            selected_node = None
            st.session_state[selected_node_key] = None
    else:
        node_options = [''] + all_nodes[:100]
        default_index = 0
        if st.session_state[selected_node_key] in node_options:
            default_index = node_options.index(st.session_state[selected_node_key])
        selected_node = st.sidebar.selectbox(
            "노드 선택", 
            options=node_options, 
            index=default_index,
            key=f"{network_type}_node_select_main"
        )
        if selected_node == '':
            selected_node = None
        st.session_state[selected_node_key] = selected_node
    
    selected_node = st.session_state[selected_node_key]
    
    # 필터 옵션
    st.sidebar.subheader(f"📊 {network_type} 필터")
    max_nodes_key = f"{network_type}_max_nodes"
    # slider는 key를 통해 자동으로 session_state와 동기화되므로 value에 session_state 사용 불필요
    max_nodes = st.sidebar.slider(
        "최대 노드 수", 
        min_value=50, 
        max_value=500, 
        value=200,  # 기본값만 설정
        step=50,
        key=max_nodes_key
    )
    
    layout_method_key = f"{network_type}_layout"
    if layout_method_key not in st.session_state:
        st.session_state[layout_method_key] = 'spring'
    layout_options = ['spring', 'pca', 'tsne']
    try:
        default_index = layout_options.index(st.session_state[layout_method_key])
    except ValueError:
        default_index = 0
        st.session_state[layout_method_key] = 'spring'
    layout_method = st.sidebar.selectbox(
        "3D 레이아웃 방법",
        options=layout_options,
        index=default_index,
        key=layout_method_key
    )
    # selectbox는 key를 통해 자동으로 session_state와 동기화되므로 수동 할당 불필요
    
    # 메인 영역
    st.subheader(f"{network_type} Network Visualization")
    
    # 네트워크 통계
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("노드 수", f"{len(graph.nodes()):,}")
    with stats_col2:
        st.metric("엣지 수", f"{len(graph.edges()):,}")
    with stats_col3:
        st.metric("평균 연결도", f"{2*len(graph.edges())/len(graph.nodes()):.2f}")
    with stats_col4:
        st.metric("네트워크 밀도", f"{nx.density(graph):.4f}")
    
    # 시각화 모드 선택
    viz_mode = st.radio(
        "시각화 모드",
        options=["3D Visualization", "2D Interactive"],
        horizontal=True,
        key=f"{network_type}_viz_mode"
    )
    
    selected_nodes = [selected_node] if selected_node else None
    
    if viz_mode == "3D Visualization":
        # 3D 네트워크 시각화
        with st.spinner("3D 네트워크를 생성하는 중..."):
            fig_3d, displayed_nodes = create_3d_network(
                graph, centrality_df, selected_nodes, max_nodes, 
                layout_method=layout_method, network_type=network_type
            )
        st.plotly_chart(fig_3d, config={'displayModeBar': True, 'displaylogo': False, 'responsive': True})
        st.caption(f"표시된 노드: {len(displayed_nodes)}개 (전체 {len(graph.nodes())}개 중)")
    else:
        # 2D 인터랙티브 네트워크
        html_string, displayed_nodes = create_interactive_network(
            graph, centrality_df, selected_nodes, max_nodes, network_type=network_type
        )
        st.components.v1.html(html_string, height=600, scrolling=True)
        st.caption(f"표시된 노드: {len(displayed_nodes)}개 (전체 {len(graph.nodes())}개 중)")
    
    # 노드 정보 및 통계
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 노드 정보")
        
        if selected_node:
            node_info = get_node_info(graph, selected_node, centrality_df)
            
            if node_info:
                st.markdown(f"### {node_info['node']}")
                
                if 'centrality' in node_info:
                    st.markdown("#### 중심성 지표")
                    metrics = node_info['centrality']
                    
                    st.markdown(f"""
                    <div class="metric-box">
                        <strong>Degree:</strong> {metrics['degree']:.4f}
                    </div>
                    <div class="metric-box">
                        <strong>Betweenness:</strong> {metrics['betweenness']:.4f}
                    </div>
                    <div class="metric-box">
                        <strong>Closeness:</strong> {metrics['closeness']:.4f}
                    </div>
                    <div class="metric-box">
                        <strong>Eigenvector:</strong> {metrics['eigenvector']:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"#### 연결된 노드 ({len(node_info['neighbors'])}개)")
                
                if node_info['edges']:
                    edges_df = pd.DataFrame(node_info['edges'])
                    edges_df = edges_df.sort_values('score', ascending=False)
                    st.dataframe(edges_df, width='stretch', height=300)
                else:
                    st.info("연결된 노드가 없습니다.")
        else:
            st.info("👈 사이드바에서 노드를 선택하세요.")
    
    with col2:
        st.subheader("📈 중심성 분포")
        
        if centrality_df is not None and not centrality_df.empty:
            # 네트워크 타입에 따른 색상 설정
            if network_type == 'PPI':
                bar_color = 'rgb(34, 139, 34)'  # 포레스트 그린
            else:
                bar_color = 'rgb(255, 215, 0)'  # 골드
            
            # Betweenness Centrality 차트
            top_betweenness = centrality_df.nlargest(20, 'Betweenness')['Betweenness']
            fig_betweenness = go.Figure(data=[
                go.Bar(
                    x=top_betweenness.index,
                    y=top_betweenness.values,
                    marker_color=bar_color,
                    text=top_betweenness.values,
                    texttemplate='%{text:.4f}',
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Betweenness: %{y:.4f}<extra></extra>'
                )
            ])
            fig_betweenness.update_layout(
                title="Top 20 Betweenness Centrality",
                xaxis_title="Node",
                yaxis_title="Betweenness Centrality",
                height=250,
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            fig_betweenness.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_betweenness, config={'displayModeBar': False, 'responsive': True})
            
            # Degree Centrality 차트
            top_degree = centrality_df.nlargest(20, 'Degree')['Degree']
            fig_degree = go.Figure(data=[
                go.Bar(
                    x=top_degree.index,
                    y=top_degree.values,
                    marker_color=bar_color,
                    text=top_degree.values,
                    texttemplate='%{text:.4f}',
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Degree: %{y:.4f}<extra></extra>'
                )
            ])
            fig_degree.update_layout(
                title="Top 20 Degree Centrality",
                xaxis_title="Node",
                yaxis_title="Degree Centrality",
                height=250,
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            fig_degree.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_degree, config={'displayModeBar': False, 'responsive': True})
            
            # 상위 중심성 노드 목록
            st.markdown("#### 상위 Betweenness Centrality 노드")
            top_nodes = centrality_df.nlargest(10, 'Betweenness')[['Betweenness', 'Degree', 'Closeness']]
            st.dataframe(top_nodes, width='stretch', height=300)

def main():
    st.markdown('<h1 class="main-header">🕸️ Autophagy Biological Network Analysis Viewer</h1>', unsafe_allow_html=True)
    
    # 사이드바
    st.sidebar.header("⚙️ 설정")
    
    # 데이터셋 모드 선택 (Research 또는 Total)
    if 'dataset_mode' not in st.session_state:
        st.session_state.dataset_mode = 'autophagy'
    
    dataset_mode = st.sidebar.radio(
        "데이터셋 선택",
        options=['Total', 'Research'],
        index=0 if st.session_state.dataset_mode == 'autophagy' else 1,
        key='dataset_mode_selector'
    )
    
    # 모드에 따라 prefix 설정
    if dataset_mode == 'Total':
        prefix = 'autophagy'
        mode_display = 'Total'
    else:
        prefix = 'research_autophagy'
        mode_display = 'Research'
    
    # 모드가 변경되면 session state 업데이트 및 캐시 클리어
    if st.session_state.dataset_mode != prefix:
        st.session_state.dataset_mode = prefix
        # 캐시 클리어
        load_network_data.clear()
        load_centrality_data.clear()
    
    # 데이터 로드
    with st.spinner(f"{mode_display} 네트워크 데이터를 로드하는 중..."):
        network_data = load_network_data(prefix=prefix)
    
    if network_data is None:
        # 폴더 경로 설정
        if prefix == 'research_autophagy':
            folder = 'research'
            ppi_file = os.path.join(folder, f'{prefix}_protein_ppi_network_edgelist.csv')
        else:
            folder = 'all'
            ppi_file = os.path.join(folder, f'{prefix}_ppi_network_edgelist.csv')
        ggi_file = os.path.join(folder, f'{prefix}_gene_network_edgelist.csv')
        st.error(f"네트워크 데이터를 로드할 수 없습니다. 다음 파일들이 있는지 확인해주세요:\n\n"
                 f"- `{ppi_file}`\n"
                 f"- `{ggi_file}`")
        return
    
    # 탭으로 네트워크 분리
    tab1, tab2 = st.tabs(["🔬 PPI Network", "🧬 GGI Network"])
    
    with tab1:
        render_network_tab('PPI', network_data, 'selected_node_ppi', prefix=prefix)
    
    with tab2:
        render_network_tab('GGI', network_data, 'selected_node_ggi', prefix=prefix)

if __name__ == "__main__":
    main()

