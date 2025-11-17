# Autophagy
Autophagy realated PPI and GGI network analysis


# Autophagy Biological Network Analysis Viewer

인터랙티브한 네트워크 분석 웹 뷰어입니다. PPI와 GGI 네트워크를 2D/3D로 시각화하고, 노드를 클릭하여 상세 정보를 확인할 수 있습니다. 
- Total:전체 오토파지 데이터
- Research: 연구에 활용된 오토파지 데이터


## 주요 기능

### 1. 데이터셋 선택
- 사이드바에서 **Total** 또는 **Research** 데이터셋을 선택할 수 있습니다.
  - **Total**: `autophagy_` 접두사 파일 사용
  - **Research**: `research_autophagy_` 접두사 파일 사용

### 2. 네트워크 탭 분리
- **PPI Network** 탭과 **GGI Network** 탭으로 분리되어 각각 독립적으로 분석 가능
- 각 탭에서 별도의 노드 검색 및 필터링 가능

### 3. 2D/3D 시각화 모드
- **3D Visualization**: Plotly를 사용한 인터랙티브 3D 네트워크 시각화
  - 레이아웃 방법 선택: Spring, PCA, t-SNE
  - 마우스로 회전, 확대/축소 가능
  - 축 레이블 없이 깔끔한 3D 공간 표시
- **2D Interactive**: Pyvis를 사용한 2D 인터랙티브 네트워크
  - 노드를 드래그하여 이동 가능
  - 초기 안정화 최적화로 움직임 최소화
  - 노드에 마우스를 올리면 중심성 정보 툴팁 표시

### 4. 색상 구분
- **PPI 네트워크**: 파란색 계열 (파란색 → 청록색 → 시안색)
- **GGI 네트워크**: 빨간색/주황색 계열 (빨간색 → 주황색 → 노란색)
- 선택된 노드: 금색으로 강조 표시
- 노드 색상은 Betweenness Centrality에 따라 그라데이션 적용

### 5. 노드 검색
- 각 네트워크 탭에서 노드 이름으로 검색 가능
- 검색된 노드는 그래프에서 강조 표시 (금색)
- 선택된 노드와 연결된 노드들도 함께 표시

### 6. 필터 옵션
- 최대 노드 수 조절 (50-500개)
- 3D 레이아웃 방법 선택 (Spring, PCA, t-SNE)
- 각 네트워크별로 독립적인 필터 설정

### 7. 노드 정보 패널
- 선택된 노드의 중심성 지표 (Degree, Betweenness, Closeness, Eigenvector)
- 연결된 노드 목록 및 연결 강도 (score)
- 상위 중심성 노드 목록

### 8. 중심성 분포 차트
- Top 20 Betweenness Centrality 차트
- Top 20 Degree Centrality 차트
- 상위 중심성 노드 데이터프레임

## 필요한 파일

웹 앱이 정상 작동하려면 다음 파일들이 필요합니다:

### Total 데이터셋
1. `autophagy_ppi_network_edgelist.csv` - PPI 네트워크 엣지 리스트
2. `autophagy_gene_network_edgelist.csv` - GGI 네트워크 엣지 리스트
3. `autophagy_PPI_centrality_analysis.csv` - PPI 중심성 분석 결과 (선택사항, 없으면 자동 계산)
4. `autophagy_GGI_centrality_analysis.csv` - GGI 중심성 분석 결과 (선택사항, 없으면 자동 계산)

### Research 데이터셋
1. `research_autophagy_protein_ppi_network_edgelist.csv` - PPI 네트워크 엣지 리스트
2. `research_autophagy_gene_network_edgelist.csv` - GGI 네트워크 엣지 리스트
3. `research_autophagy_PPI_centrality_analysis.csv` - PPI 중심성 분석 결과 (선택사항, 없으면 자동 계산)
4. `research_autophagy_GGI_centrality_analysis.csv` - GGI 중심성 분석 결과 (선택사항, 없으면 자동 계산)

**참고**: 중심성 분석 파일이 없으면 자동으로 계산되지만, 시간이 오래 걸릴 수 있습니다. 미리 계산된 파일을 사용하는 것을 권장합니다.

## 사용 팁

- **데이터셋 전환**: 사이드바에서 Total과 Research를 쉽게 전환할 수 있습니다.
- **성능 최적화**: 큰 네트워크의 경우 최대 노드 수를 조절하여 성능을 최적화할 수 있습니다.
- **노드 선택**: 노드를 선택하면 해당 노드와 연결된 노드들도 함께 표시됩니다.
- **3D 시각화**: 3D 모드에서는 마우스로 그래프를 회전하고 확대/축소할 수 있습니다.
- **2D 시각화**: 2D 모드에서는 노드를 드래그하여 이동할 수 있으며, 초기 안정화가 최적화되어 움직임이 최소화됩니다.
- **색상 구분**: PPI는 파란색 계열, GGI는 빨간색/주황색 계열로 구분되어 시각적으로 쉽게 구분할 수 있습니다.
- **레이아웃 선택**: 3D 시각화에서 Spring, PCA, t-SNE 레이아웃을 선택하여 다양한 관점에서 네트워크를 관찰할 수 있습니다.

