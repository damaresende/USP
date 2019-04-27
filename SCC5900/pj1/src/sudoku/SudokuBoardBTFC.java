package sudoku;

import java.util.LinkedList;

public class SudokuBoardBTFC extends SudokuBoardBT {
	LinkedList<Integer> [] domain;
	
	public SudokuBoardBTFC() {
		super();
		domain = (LinkedList<Integer>[]) new LinkedList<?>[boardSize * boardSize];
		for(int i = 0; i < boardSize * boardSize; i++) {
			domain[i] = new LinkedList<Integer>();
		}
	}
	
	public LinkedList<Integer> getDomain(int i, int j) {
		return domain[i * boardDim + j];
	}
	
	public LinkedList<Integer> getDomain(int k) {
		return domain[k];
	}
	
	protected void initDomain() {
		for(int i = 0; i < boardSize; i++) {
			for (int j = 0; j < boardSize; j++) {
				if (board[i][j] != 0) {
					domain[i * boardSize + j].add(board[i][j]);
				} else {
					for (int k = 1; k <= boardSize; k++) {
						if (evaluate(new Coordinates(i, j), k))
							domain[i * boardSize + j].add(k);
					}
				}
			}
		}
	}
	
	protected void updateDomain(int i, int j, int value) {
		LinkedList<Integer> domain = getDomain(i, j);
		domain.clear();
		domain.add(value);
		
		for (int k = i + 1; k < boardSize; k++) {
			domain = getDomain(k * boardSize + j);
			int d = domain.indexOf(value);
			
			if (d > 0)
				domain.remove(d);
		}
		
		for (int k = j + 1; k < boardSize; k++) {
			domain = getDomain(i * boardSize + k);
			int d = domain.indexOf(value);
			
			if (d > -1)
				domain.remove(d);
		}
	}
	
	protected void restoreDomain(int i, int j, int value) {
		
	}
}
