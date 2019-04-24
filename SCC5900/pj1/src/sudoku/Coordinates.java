/*
 * Base object of a Sudoku board. It contains a cell location and 
 * square region pivot. Only 9 x 9 boards are valid.
 * 
 * @author: Damares Resende
 * @contact: damaresresende@usp.br
 * @since: Apr 14, 2019
 * 
 * @organization: University of Sao Paulo (USP)
 *     Institute of Mathematics and Computer Science (ICMC)
 *     Project of Algorithms Class (SCC5000)
*/

package sudoku;

public class Coordinates {
	private int i = -1, j = -1, pi = -1, pj = -1;

	/**
     * Sets coordinate values and pivot
     * 
     * @param i: row coordinate
     * @param j: column coordinate
     */
	public Coordinates(int i, int j) {
		this.i = i;
		this.j = j;
		setPivots();
	}
	
	/**
     * Sets pivot values based on the coordinates of the cell
     */
	private void setPivots() {
		int boardDim = 3;
		float boardSize = boardDim * boardDim;
		int[][] pivots = {{0, 0}, {0, 3}, {0, 6}, {3, 0}, {3, 3}, {3, 6}, {6, 0}, {6, 3}, {6, 6}};
		
		int i = boardDim - Math.round(boardSize/(this.i+1));
		if (i < 0) i = 0;
		
		int j = boardDim - Math.round(boardSize/(this.j+1));
		if (j < 0) j = 0;
		
		int group = i * boardDim + j;
		this.pi = pivots[group][0];
		this.pj = pivots[group][1];
	}
	
	/**
     * Retrieves row coordinate
     * 
     * @return integer with row coordinate
     */
	public int getI() {
		return this.i;
	}
	
	/**
     * Retrieves column coordinate
     * 
     * @return integer with column coordinate
     */
	public int getJ() {
		return this.j;
	}
	
	/**
     * Retrieves pivot row coordinate
     * 
     * @return integer with pivot row coordinate
     */
	public int getPivotI() {
		return this.pi;
	}
	
	/**
     * Retrieves pivot column coordinate
     * 
     * @return integer with pivot column coordinate
     */
	public int getPivotJ() {
		return this.pj;
	}
}
