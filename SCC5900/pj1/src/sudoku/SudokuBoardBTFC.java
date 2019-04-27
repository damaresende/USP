package sudoku;

import java.util.LinkedList;

public class SudokuBoardBTFC extends SudokuBoardBT {
	LinkedList<Integer> [] domain;
	int [] backup;
	int [] cellBackup;
	
	public SudokuBoardBTFC() {
		super();
		domain = (LinkedList<Integer>[]) new LinkedList<?>[boardSize * boardSize];
		backup = new int[boardSize * boardSize];
		cellBackup = new int[boardSize];
		
		for(int i = 0; i < boardSize * boardSize; i++) {
			domain[i] = new LinkedList<Integer>();
		}
	}
	
	public LinkedList<Integer> getDomain(int i, int j) {
		return domain[i * boardSize + j];
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
	
	private void prepareBackup(int i, int j, LinkedList<Integer> dom) {
		for (int k = 0; k < boardSize * boardSize; k++) {
			backup[k] = 0;
		}
		
		for (int k = 0; k < boardSize; k++) {
			cellBackup[k] = 0;
		}
		
		for (Integer value : dom) {
			cellBackup[value - 1] = 1;
		}
		
	}
	
	protected void updateDomain(int i, int j, int value) {
		LinkedList<Integer> dom = getDomain(i, j);
		
		prepareBackup(i, j, dom);
		
		dom.clear();
		dom.add(value);
		
		for (int k = i + 1; k < boardSize; k++) {
			dom = getDomain(k * boardSize + j);
			
			int d = dom.indexOf(value);
			
			if (d > 0) {
				dom.remove(d);
				backup[k * boardSize + j] = 1;
			}
		}
		
		for (int k = j + 1; k < boardSize; k++) {
			dom = getDomain(i * boardSize + k);
			
			int d = dom.indexOf(value);
			
			if (d > -1) {
				dom.remove(d);
				backup[i * boardSize + k] = 1;
			}
		}
		
		Coordinates cell = new Coordinates(i, j);
		for (int x = i; x < cell.getPivotI() + boardDim; x++) {
			for (int y = j; y < cell.getPivotJ() + boardDim; y++) {
				if (x != i && y != j) {
					dom = getDomain(x * boardSize + y);
					
					int d = dom.indexOf(value);
					
					if (d > -1) {
						dom.remove(d);
						backup[x * boardSize + y] = 1;
					}
				}
			}
		}
	}
	
	protected void restoreDomain(int i, int j, int value) {
		LinkedList<Integer> dom = getDomain(i, j);
		
		dom.removeFirst();
		for(int k = 0; k < boardSize; k++)
			if (cellBackup[k] == 1)
				dom.add(k + 1);
		
		for (int k = 0; k < boardSize * boardSize; k++) {
			if (backup[k] == 1) {
				domain[k].add(value);
			}
		}
			
	}
}
