package sudoku;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class SudokuBoardBTFC extends SudokuBoardBT {
	boolean [][] domain;
	
	public SudokuBoardBTFC() {
		super();
		domain = new boolean[boardSize * boardSize][boardSize];
		
		for (int i = 0; i < boardSize * boardSize; i++)
			for (int j = 0; j < boardSize; j++)
				domain[i][j] = true;
	}
	
	public boolean[] getDomain(int i, int j) {
		boolean [] cellDomain = new boolean[boardSize];
		for(int k = 0; k < boardSize; k++) {
			cellDomain[k] = domain[i * boardDim + j][k];
		}
		return cellDomain;
	}
	
	private void removeFromDomain(int ii, int jj, int value) {
		for (int k = 0; k < boardSize; k++)
			if (board[ii][k] != value)
				domain[ii * boardSize + k][value-1] = false;
		
		for (int k = 0; k < boardSize; k++)
			if (board[k][jj] != value)
				domain[k * boardSize + jj][value-1] = false;
	}
	
	@Override
	public boolean fillData(String boardFile) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(boardFile));
			String line = "";
			int i = 0;
			
			while ((line = reader.readLine()) != null) {
				String [] parts = line.trim().split(" ");
				
				if (parts.length != boardSize) {
					System.out.println("ERROR: The number of board values per row should be "  
							+ String.valueOf(boardSize) + ".");
					reader.close();
					return false;
				}
				
				for (int j = 0; j < boardSize; j++) {
					if (parts[j].equals("-")) {
						board[i][j] = -1;
						toComplete.add(new Coordinates(i, j));
					}
					else {
						board[i][j] = Integer.parseInt(parts[j]);
						removeFromDomain(i, j, board[i][j]);
						
						if (board[i][j] < 0) {
							System.out.println("ERROR: board values must be greater than 0.");
							reader.close();
							return false;
						}
					}
				}
				i++;
			}
			reader.close();
		} catch (IOException e) {
			System.out.println("ERROR: Board file could not be read");
			return false;
		} catch (NumberFormatException e) {
			System.out.println("ERROR: Board value could not be parsed");
			return false;
		}
		return true;
	}
	
}
