package sudoku;

public class Coordinates {
	private int i = -1, j = -1, pi = -1, pj = -1;
	private int boardDim = 3;
	private float boardSize = boardDim * boardDim;
	private int[][] pivots = {{0, 0}, {0, 3}, {0, 6}, {3, 0}, {3, 3}, {3, 6}, {6, 0}, {6, 3}, {6, 6}};
	
	public Coordinates(int i, int j) {
		this.i = i;
		this.j = j;
		setPivots();
	}
	
	private void setPivots() {
		int i = boardDim - Math.round(boardSize/(this.i+1));
		if (i < 0) i = 0;
		
		int j = boardDim - Math.round(boardSize/(this.j+1));
		if (j < 0) j = 0;
		
		int group = i * boardDim + j;
		this.pi = pivots[group][0];
		this.pj = pivots[group][1];
	}
	
	public int getI() {
		return this.i;
	}
	
	public int getJ() {
		return this.j;
	}
	
	public int getPivotI() {
		return this.pi;
	}
	
	public int getPivotJ() {
		return this.pj;
	}
}
