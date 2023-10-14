import React, { useEffect, useRef, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import param from './weights_and_biases_3_16.json'

type NNLayer = {
  weights: number[][];
  biases: number[];
};

type NNParams = {
  layer_1: NNLayer;
  layer_2: NNLayer;
  layer_3: NNLayer;
  // layer_4: NNLayer;
  // layer_5: NNLayer;
  // Add more layers as needed
};


function relu(x: number) {
  return Math.max(0, x);
}

function softmax(arr: number[]) {
  return arr.map(function (value, index) {
    return Math.exp(value) / arr.map(function (y /*value*/) { return Math.exp(y) }).reduce(function (a, b) { return a + b })
  })
}

function forwardPass(input: number[], weights: number[][], biases: number[], activation: string) {
  let output = [];

  for (let i = 0; i < weights[0].length; i++) {
    output[i] = input.reduce((acc, val, j) => {
      return acc + val * weights[j][i]
    }) + biases[i];


    if (activation === 'relu') {
      output[i] = relu(output[i]);
    }
  }
  if (activation === 'softmax') {
    output = softmax(output);
  }
  return output;
}

function fillColor(percent: number, ctx: CanvasRenderingContext2D) {
  let cor = 1 - percent
  return ctx.fillStyle = `rgb(${256 * cor},${256 * cor},${256 * cor})`
}

const percScreen = 0.80
const gridSize = 28

function App() {
  const circlesPerRow = [16, 16, 10];
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [windowSize, setWindowSize] = useState({ width: window.innerHeight * percScreen, height: window.innerHeight * percScreen });
  const [grid, setGrid] = useState<number[][]>(Array.from({ length: gridSize }, () => Array(gridSize).fill(0)))
  const [drawing, setDrawing] = useState(false);
  const [params, setParams] = useState<NNParams>(param);
  const [circleSize, setCircleSize] = useState<string>(`${((window.innerHeight * percScreen) / (Math.max(...circlesPerRow) + Math.max(...circlesPerRow) * 0.5 - 0.5))}px`)
  const [gapSize, setGapSize] = useState<string>(`${((window.innerHeight * percScreen) / (2 * Math.max(...circlesPerRow) + Math.max(...circlesPerRow) - 1))}px`)
  const [circleConditions, setCircleConditions] = useState<number[]>(Array(circlesPerRow.reduce((acc, val) => acc + val)).fill(0))
  const [nodeVale, setNodeValue] = useState<number[]>(Array(circlesPerRow.reduce((acc, val) => acc + val)).fill(0))
  const [predictedValue, setPredictedValue] = useState<string>('-')

  function predict(input: number[], params: NNParams) {

    let result = input;

    let result1 = forwardPass(result, params.layer_1.weights, params.layer_1.biases, 'relu');

    let result2 = forwardPass(result1, params.layer_2.weights, params.layer_2.biases, 'relu');

    let result3 = forwardPass(result2, params.layer_3.weights, params.layer_3.biases, 'softmax');

    let maxResult1 = Math.max(...result1)
    let maxResult2 = Math.max(...result2)

    setNodeValue([...result1, ...result2, ...result3])
    setCircleConditions([...result1.map(a => a / maxResult1), ...result2.map(a => a / maxResult2), ...result3])

    return result3;
  }

  useEffect(() => {

    function handleResize() {
      setWindowSize({
        width: window.innerHeight * percScreen,
        height: window.innerHeight * percScreen
      });
    }

    window.addEventListener('resize', handleResize);
    handleResize(); // initialize size
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {

    var flattenedGrid = grid.flat();
    if (!params) return
    predict(flattenedGrid, params)


  }, [grid]);


  useEffect(() => {
    setGrid(Array.from({ length: gridSize }, () => Array(gridSize).fill(0)));

    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = windowSize.width;
    canvas.height = windowSize.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    for (let x = 0; x <= canvas.width; x += canvas.width / gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
    }
    for (let y = 0; y <= canvas.height; y += canvas.height / gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
    }
    ctx.strokeStyle = 'black';
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.stroke();

    setCircleSize(`${((window.innerHeight * percScreen) / (Math.max(...circlesPerRow) + Math.max(...circlesPerRow) * 0.5 - 0.5))}px`)
    setGapSize(`${((window.innerHeight * percScreen) / (2 * Math.max(...circlesPerRow) + Math.max(...circlesPerRow) - 1))}px`)
  }, [windowSize]);


  const renderCircles = (circleArray: number[], column: number) => {
    let addNumber = 0
    if (column > 0) {
      //So our circle index stays consistent with circleConditions
      addNumber += circleArray.slice(0, column).reduce((acc, val) => acc + val)
    }


    return Array.from({ length: circleArray[column] }, (_, index) => {
      return <div key={index + addNumber} style={{
        height: circleSize,
        width: circleSize,
        backgroundColor: `rgba(255,255,255,${circleConditions[index + addNumber]})`
      }} className="circle"
      title={`${nodeVale[index + addNumber]}`} />
    }
    )
  };

  const renderLines = (prevLayerSize: number, nextLayerSize: number) => {
    const circleRadius = parseInt(circleSize) / 2;
    const circleGap = parseInt(gapSize);
    
    // Calculate the total height occupied by circles and gaps for both layers
    const prevLayerHeight = prevLayerSize * circleRadius * 2 + (prevLayerSize - 1) * circleGap;
    const nextLayerHeight = nextLayerSize * circleRadius * 2 + (nextLayerSize - 1) * circleGap;
    
    // Determine the starting y-coordinates for both layers
    const startY = (windowSize.height - prevLayerHeight) / 2 + circleRadius;
    const endY = (windowSize.height - nextLayerHeight) / 2 + circleRadius;
    
    let svgLines = [];

    let someNumber = 0
    
    for (let i = 0; i < prevLayerSize; i++) {
      for (let j = 0; j < nextLayerSize; j++) {
        svgLines.push(
          <line
            x1="0"
            y1={startY + i * (circleRadius * 2 + circleGap)}
            x2="100%"
            y2={endY + j * (circleRadius * 2 + circleGap)}
            stroke="white"
            stroke-width="0.3"
          />
        );
        someNumber += 1
        console.log(someNumber)
      }
    }
    
    return (
      <div style={{ width: "10vw", height: "100%" }}>
        <svg width="100%" height="100%">
          {svgLines}
        </svg>
      </div>
    );
  }
  
  const renderRows = (circlesArray: number[]) => (
    circlesArray.map((_, index) => (
      <>
        {/* {index != 0 &&
          renderLines(circlesArray[index-1], circlesArray[index])
        } */}
        <div key={index} style={{ gap: gapSize }} className="row">
          {renderCircles(circlesArray, index)}
        </div>
      </>
    ))
  );

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setDrawing(true);
  }

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx || !drawing) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const gridSize = 28;
    const cellX = Math.floor(x / (windowSize.width / gridSize));
    const cellY = Math.floor(y / (windowSize.height / gridSize));



    const increaseVal = 0.33333


    if (grid[cellY][cellX] != 1) {
      const newGrid = [...grid.map(row => [...row])];



      // The action you want to perform
      const performAction = (y: number, x: number) => {
        if (newGrid[y][x] < 1 - increaseVal) {
          newGrid[y][x] += increaseVal;
        } else {
          newGrid[y][x] = 1;
        }
        ctx.fillStyle = fillColor(newGrid[y][x], ctx);
        ctx.fillRect(x * (windowSize.width / gridSize), y * (windowSize.height / gridSize), windowSize.width / gridSize, windowSize.height / gridSize);
      };

      const performAdjacentAction = (y: number, x: number, diff: number) => {
        if (newGrid[y][x] < diff) {
          newGrid[y][x] = diff

          ctx.fillStyle = fillColor(newGrid[y][x], ctx);
          ctx.fillRect(x * (windowSize.width / gridSize), y * (windowSize.height / gridSize), windowSize.width / gridSize, windowSize.height / gridSize);
        }


      };

      performAction(cellY, cellX)

      // Offsets for the 8 surrounding cells
      const offsets = [
        [-1, 0],
        [0, -1], [0, 1],
        [1, 0]
      ];

      const secondaryOffset = [
        [-1, -1], [-1, 1],

        [1, -1], [1, 1]
      ]

      offsets.forEach(([yOffset, xOffset]) => {
        const newY = cellY + yOffset;
        const newX = cellX + xOffset;

        // Check if the cell is within the grid bounds
        if (newY >= 0 && newY < grid.length && newX >= 0 && newX < grid[0].length) {
          performAdjacentAction(newY, newX, newGrid[cellY][cellX] - 0.1);
        }
      });

      secondaryOffset.forEach(([yOffset, xOffset]) => {
        const newY = cellY + yOffset;
        const newX = cellX + xOffset;

        // Check if the cell is within the grid bounds
        if (newY >= 0 && newY < grid.length && newX >= 0 && newX < grid[0].length) {
          performAdjacentAction(newY, newX, newGrid[cellY][cellX] - 0.2);
        }
      });

      setGrid(newGrid);
    }
  }

  const clearCanvasAndGrid = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // prevent the browser's context menu from appearing

    const canvas = canvasRef.current;
    if (!canvas) return
    const ctx = canvas?.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Redraw the grid lines if necessary (if you want to maintain the grid lines visible)
    const gridSize = 28;
    for (let x = 0; x <= canvas.width; x += canvas.width / gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
    }
    for (let y = 0; y <= canvas.height; y += canvas.height / gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
    }
    ctx.strokeStyle = 'black';
    ctx.stroke();

    // Clear grid
    setGrid(Array.from({ length: gridSize }, () => Array(gridSize).fill(0)));
  };

  const endDrawing = () => {
    if (!drawing) return
    setDrawing(false);

    var flattenedGrid = grid.flat();
    if (!params) return
    var predicted = predict(flattenedGrid, params)


    const maxIndex = predicted.reduce((maxIndex, currentValue, currentIndex, array) => {
      return currentValue > array[maxIndex] ? currentIndex : maxIndex;
    }, 0);

    setPredictedValue(predicted[maxIndex] > 0.8 ? `${maxIndex}` : '-')


    for (let i = 0; i < grid.length; i++) {
      var printyThing = ' '
      for (let j = 0; j < grid[i].length; j++) {
        printyThing += grid[i][j] == 1 ? `1.0 ` : grid[i][j] == 0 ? `0.0 ` : `${grid[i][j]} `
      }
      console.log(printyThing)
    }
    console.log('')
  };




  return (
    <div className="App">
      <header className="App-header">
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={endDrawing}
          onMouseLeave={endDrawing}
          onContextMenu={clearCanvasAndGrid}
        />
        <div className="NN-layer" style={{ height: `${windowSize.height}px` }}>
          {renderRows(circlesPerRow)}
        </div>
        <div style={{ width: "40px" }}>
          <h1>
            {predictedValue}
          </h1>
        </div>
      </header>
    </div>
  );
}

export default App;

