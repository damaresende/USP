/*
 * Creates a Sudoku board based on a matrix in a text file and uses
 * backtracking to solve it.
 * 
 * @author: Damares Resende
 * @contact: damaresresende@usp.br
 * @since: Apr 24, 2019
 * 
 * @organization: University of Sao Paulo (USP)
 *     Institute of Mathematics and Computer Science (ICMC)
 *     Project of Algorithms Class (SCC5000)
*/

package sudoku;

import java.util.LinkedList;

public class SudokuBoardBTFC extends SudokuBoardBT {
	int [] backup;
	int [] cellBackup;
	LinkedList<Integer> [] domain;
	
	/**
     * Creates a dim x dim matrix
     * 
     * @param dim: dimension of the Sudoku board
     */
	public SudokuBoardBTFC(int dim) {
		super(dim);
		domain = (LinkedList<Integer>[]) new LinkedList<?>[boardSize * boardSize];
		backup = new int[boardSize * boardSize];
		cellBackup = new int[boardSize];
		
		for(int i = 0; i < boardSize * boardSize; i++) {
			domain[i] = new LinkedList<Integer>();
		}
	}
	
	/**
     * Retrieves domain list for the specified coordinates
     * 
     * @param i: board row
     * @param i: board column
     */
	public LinkedList<Integer> getDomain(int i, int j) {
		return domain[i * boardSize + j];
	}
	
	/**
     * Retrieves domain list for the specified row index
     * 
     * @param k: index from 0 to dim*dim
     */
	public LinkedList<Integer> getDomain(int k) {
		return domain[k];
	}
	
	/**
     * Initializes domain based on the board loaded
     */
	protected void initDomain() {
		for(int i = 0; i < boardSize * boardSize; i++) {
			domain[i].clear();
		}
		
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
	
	/**
     * Cleans backup arrays for the next iteration
     * 
     * @param i: board row
     * @param i: board column
     * @param dom: domain to have values backed up
     */
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

	/**
     * Updates domain based on the current cell attribution. It first cleans
     * the current cell domain, then all cells in the same column, all
     * cells in the same row and finally all cells in the same square
     * 
     * @param i: board row
     * @param j: board column
     * @param value: value to set cell to
     */
	protected void updateDomain(int i, int j, int value) {
		LinkedList<Integer> dom = getDomain(i, j);
		prepareBackup(i, j, dom);
		
		dom.clear();
		dom.add(value);
		
		for (int k = i + 1; k < boardSize; k++) {
			dom = getDomain(k * boardSize + j);
			
			int d = dom.indexOf(value);
			
			if (d > -1) {
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
		for (int x = cell.getPivotI(); x < cell.getPivotI() + boardDim; x++) {
			for (int y = cell.getPivotJ(); y < cell.getPivotJ() + boardDim; y++) {
				if (x  * boardSize + y > i  * boardSize + j) {
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
	
	/**
     * Restores domain based on the data stored in the backup arrays.
     * It first restores the domain of the cell that was modified and
     * then it restores the domain of all the other cells related to 
     * that change
     * 
     * @param i: board row
     * @param j: board column
     * @param value: value that cell was set to
     */
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
	
	/**
     * Applies backtracking algorithm with forward checking (not working =/)
     * 
     * @param cell: cell to start backtracking with
     */
	public boolean backtracking(Coordinates cell) {
		LinkedList<Integer> dom = getDomain(cell.getI(), cell.getJ());
		
		while(dom.size() > 0) {
			int value = dom.removeFirst();
					
			board[cell.getI()][cell.getJ()] = value;
			updateDomain(cell.getI(), cell.getJ(), value);
			
			if (toComplete.size() == 0) {
				return true;
			}
			
			if (checkForward(cell.getI(), cell.getJ())) {
				if (backtracking(toComplete.poll())) {
					return true;
				}
			} else {
				dom.add(value);
				board[cell.getI()][cell.getJ()] = -1;
				restoreDomain(cell.getI(), cell.getJ(), value);
			}
		}
		
		toComplete.add(cell);
		return false;
	}
	
	/**
     * Validates that there are no empty domains beyond the
     * specified cell
     * 
     * @param i: board row
     * @param j: board column
     */
	public boolean checkForward(int i, int j) {
		for(int k = i * boardSize + j; k < boardSize * boardSize; k++) {
			if (getDomain(k).size() == 0)
				return false;
		}
		return true;
	}
}
