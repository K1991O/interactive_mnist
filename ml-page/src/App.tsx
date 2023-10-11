import React, { useEffect, useRef, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import param from './weights_and_biases.json'

type NNLayer = {
  weights: number[][];
  biases: number[];
};

type NNParams = {
  layer_1: NNLayer;
  layer_2: NNLayer;
  layer_3: NNLayer;
  // Add more layers as needed
};




const fetchWeightsAndBiases = async (): Promise<NNParams | null> => {
  try {
    const response = await fetch('/weights_and_biases.json');

    if (!response.ok) {
      console.error('Network response was not ok:', response);
      return null;
    }

    const text = await response.text();


    try {
      const myJson: NNParams = JSON.parse(text);
      return myJson;
    } catch (jsonError) {
      console.error('Error parsing JSON:', jsonError);
      return null;
    }

  } catch (error) {
    console.error('Error during fetch operation: ', error);
    return null;
  }
};


function relu(x: number) {
  return Math.max(0, x);
}

function softmax(arr: number[]) {
  return arr.map(function(value,index) { 
    return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
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

function predict(input: number[], params: NNParams) {
  let result = input;

  let result1 = forwardPass(result, params.layer_1.weights, params.layer_1.biases, 'relu');

  let result2 = forwardPass(result1, params.layer_2.weights, params.layer_2.biases, 'relu');

  let result3 = forwardPass(result2, params.layer_3.weights, params.layer_3.biases, 'softmax');

  return result3;
}


function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [windowSize, setWindowSize] = useState({ width: window.innerHeight * 0.65, height: window.innerHeight * 0.65 });
  const [grid, setGrid] = useState<number[][]>([])
  const [drawing, setDrawing] = useState(false);
  const [params, setParams] = useState<NNParams>(param);


  useEffect(() => {

    function handleResize() {
      setWindowSize({
        width: window.innerHeight * 0.65,
        height: window.innerHeight * 0.65
      });
    }

    window.addEventListener('resize', handleResize);
    handleResize(); // initialize size
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const gridSize = 28;
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
  }, [windowSize]);


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


    ctx.fillStyle = 'black';
    ctx.fillRect(cellX * (windowSize.width / gridSize), cellY * (windowSize.height / gridSize), windowSize.width / gridSize, windowSize.height / gridSize);

    const newGrid = [...grid];
    newGrid[cellY][cellX] = 1;

    setGrid(newGrid);
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
    console.log("end drawing")
    setDrawing(false);

    var flattenedGrid = grid.flat();
    console.log(flattenedGrid)
    if (!params) return
    var predicted = predict(flattenedGrid, params)


    const maxIndex = predicted.reduce((maxIndex, currentValue, currentIndex, array) => {
      return currentValue > array[maxIndex] ? currentIndex : maxIndex;
    }, 0); 

    console.log(maxIndex)
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
      </header>
    </div>
  );
}

export default App;

