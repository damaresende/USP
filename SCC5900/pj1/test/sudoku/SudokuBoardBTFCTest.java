/*
 * Unit tests for creating and filling up a Sudoku board
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

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;

import org.junit.BeforeClass;
import org.junit.Test;


public class SudokuBoardBTFCTest {

	public static String boardFile;
	public static SudokuBoardBT answer;
	public static SudokuBoardBTFC board;
	
	/**
     * Setting up answer board and board file name
     */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		boardFile = System.getProperty("user.dir") + File.separator + "test" + File.separator 
				+ "boards" + File.separator + "board1.txt";
		
		String answerFile = System.getProperty("user.dir") + File.separator + "test" + File.separator 
				+ "boards" + File.separator + "board1_answer.txt";
		
		String line = "";
		BufferedReader reader = new BufferedReader(new FileReader(answerFile));
		
		answer = new SudokuBoardBT(3);
		while ((line = reader.readLine()) != null) {
			answer.fillData(line);
		}
		reader.close();
		
		reader = new BufferedReader(new FileReader(boardFile));
		
		board = new SudokuBoardBTFC(3);
		while ((line = reader.readLine()) != null) {
			board.fillData(line);
		}
		reader.close();
	}
	
	/**
     * Tests if domain values are initialized to true for all values
	 * @throws IOException 
     */
	@Test
	public void testBoardInitialization() throws IOException {
		String line = "";
		BufferedReader reader = new BufferedReader(new FileReader(boardFile));
		
		SudokuBoardBTFC sboard = new SudokuBoardBTFC(3);
		while ((line = reader.readLine()) != null) {
			sboard.fillData(line);
		}
		reader.close();
		
		for(int i = 0; i < sboard.getBoardSize(); i++) {
			for (int j = 0; j < sboard.getBoardSize(); j++) {
				assertNotNull(sboard.getDomain(i, j));
				assertEquals(0, sboard.getDomain(i, j).size());
			}
		}
	}
	
	/**
     * Tests if not valid values are removed from domain
     */
	@Test
	public void testFillDataDomain() {
		board.initDomain();
		
		String initialDomain = System.getProperty("user.dir") + File.separator + "test" + File.separator 
				+ "boards" + File.separator + "initial_domain.txt";
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(initialDomain));
			String line = "";
		
			int i = 0;
			while ((line = reader.readLine()) != null) {
				String [] parts = line.trim().split(" ");
			
				for (int k = 0; k < parts.length; k++) {
					LinkedList<Integer> domain = board.getDomain(i);
					assertTrue(domain.get(k) == Integer.parseInt(parts[k]));
				}
				i++;
			}
			reader.close();
		} catch (IOException e) {
			System.out.println("ERROR: Domain file could not be read");
		}
	}

	/**
     * Tests if update domain is updating the row of the cell a value is set
     */
	@Test
	public void testUpdateDomainCellRow() {
		board.initDomain();
		board.updateDomain(5, 2, 7);
		
		assertEquals(1, board.getDomain(5, 2).size());
		assertTrue(7 == board.getDomain(5, 2).getFirst());
	}
	
	/**
     * Tests if update domain is updating the domain of cells in the
     * same column of the cell that was set
     */
	@Test
	public void testUpdateDomainCleanRows() {
		board.initDomain();	
		board.updateDomain(2, 2, 4);
		
		// row to set to one value
		assertEquals(1, board.getDomain(2, 2).size());
		assertTrue(4 == board.getDomain(2, 2).getFirst());
		
		// rows to keep
		assertEquals(2, board.getDomain(2).size());
		assertTrue(3 == board.getDomain(2).get(0));
		assertTrue(9 == board.getDomain(2).get(1));
		
		assertEquals(1, board.getDomain(11).size());
		assertTrue(8 == board.getDomain(11).get(0));
		
		// rows to clean
		int [] rowsToClean = new int[]{29, 38, 47, 54, 65, 74};
		for (int row : rowsToClean) {
			assertEquals(-1, board.getDomain(row).indexOf(4));
		}
		
		assertEquals(0, board.backup[29]);
		assertEquals(0, board.backup[38]);
		assertEquals(0, board.backup[47]);
		assertEquals(0, board.backup[54]);
		assertEquals(0, board.backup[65]);
		assertEquals(0, board.backup[74]);
	}
	
	/**
     * Tests if update domain is updating the domain of cells in the
     * same row of the cell that was set
     */
	@Test
	public void testUpdateDomainCleanColumns() {
		board.initDomain();		
		board.updateDomain(2, 2, 4);
		
		// row to set to one value
		assertEquals(1, board.getDomain(2, 2).size());
		assertTrue(4 == board.getDomain(2, 2).getFirst());
		
		// columns to keep
		assertEquals(1, board.getDomain(18).size());
		assertTrue(2 == board.getDomain(18).get(0));
		
		assertEquals(4, board.getDomain(19).size());
		assertTrue(3 == board.getDomain(19).get(0));
		assertTrue(5 == board.getDomain(19).get(1));
		assertTrue(7 == board.getDomain(19).get(2));
		assertTrue(9 == board.getDomain(19).get(3));
		
		// columns to clean
		int [] columnsToClean = new int[]{21, 22, 23, 24, 25, 26};
		for (int col : columnsToClean) {
			assertEquals(-1, board.getDomain(col).indexOf(4));
		}
		
		assertEquals(0, board.backup[21]);
		assertEquals(0, board.backup[22]);
		assertEquals(0, board.backup[23]);
		assertEquals(1, board.backup[24]);
		assertEquals(1, board.backup[25]);
		assertEquals(0, board.backup[26]);
	}
	
	/**
     * Tests if update domain is updating the domain of cells in the
     * same square of the cell that was set
     */
	@Test
	public void testUpdateDomainCleanSquare() {
		board.initDomain();		
		board.updateDomain(4, 3, 8);
		
		// row to set to one value
		assertEquals(1, board.getDomain(4, 3).size());
		assertTrue(8 == board.getDomain(4, 3).getFirst());
		
		// cells to keep
		assertEquals(1, board.getDomain(30).size());
		assertTrue(4 == board.getDomain(30).get(0));
		
		assertEquals(3, board.getDomain(31).size());
		assertTrue(2 == board.getDomain(31).get(0));
		assertTrue(3 == board.getDomain(31).get(1));
		assertTrue(5 == board.getDomain(31).get(2));
		
		assertEquals(1, board.getDomain(32).size());
		assertTrue(7 == board.getDomain(32).get(0));
		
		// cells to clean
		int [] columnsToClean = new int[]{40, 41, 48, 49, 50};
		for (int col : columnsToClean) {
			assertEquals(-1, board.getDomain(col).indexOf(8));
		}
		
		assertEquals(1, board.backup[40]);
		assertEquals(0, board.backup[41]);
		assertEquals(0, board.backup[48]);
		assertEquals(1, board.backup[49]);
		assertEquals(0, board.backup[50]);
	}
	

	/**
     * Tests if domain is correctly restored
     */
	@Test
	public void testRestoreDomain() {
		board.initDomain();
		
		board.updateDomain(5, 1, 3);
		assertEquals(1, board.getDomain(5, 1).size());
		assertTrue(3 == board.getDomain(5, 1).getFirst());
		
		board.restoreDomain(5, 1, 3);
		assertEquals(3, board.getDomain(5, 1).size());
		assertTrue(2 == board.getDomain(5, 1).get(0));
		assertTrue(3 == board.getDomain(5, 1).get(1));
		assertTrue(5 == board.getDomain(5, 1).get(2));
		
		int [] cellsToRestore = new int[]{47, 49, 55, 64};
		for (int cell : cellsToRestore) {
			assertTrue(board.getDomain(cell).indexOf(3) > -1);
		}
	}
	
//	/**
//     * Tests if all values to be filled are filled after backtracking
//	 * @throws IOException 
//     */
//	@Test
//	public void testBacktrackingFullFill() throws IOException {
//		String line = "";
//		BufferedReader reader = new BufferedReader(new FileReader(boardFile));
//		
//		SudokuBoardBTFC sboard = new SudokuBoardBTFC(3);
//		while ((line = reader.readLine()) != null) {
//			sboard.fillData(line);
//		}
//		reader.close();
//		
//		sboard.initDomain();
//		sboard.backtracking(sboard.getNextCellToFill());
//		
//		assertTrue(sboard.toCompleteIsEmpty());
//	}
//	
//	/**
//     * Tests if all values to be filled are correctly filled after backtracking
//	 * @throws IOException 
//     */
//	@Test
//	public void testBacktrackingCorrectFill() throws IOException {
//		String line = "";
//		BufferedReader reader = new BufferedReader(new FileReader(boardFile));
//		
//		SudokuBoardBTFC sboard = new SudokuBoardBTFC(3);
//		while ((line = reader.readLine()) != null) {
//			sboard.fillData(line);
//		}
//		reader.close();
//		
//		sboard.backtracking(sboard.getNextCellToFill());
//
//		for(int i = 0; i < sboard.getBoardSize(); i++) {
//			for (int j = 0; j < sboard.getBoardSize(); j++) {
//				assertEquals(answer.getCellValue(i, j), sboard.getCellValue(i, j));
//			}
//		}
//	}
}
