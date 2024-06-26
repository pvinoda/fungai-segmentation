import React, { useRef, useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';

function Canvas({ imageSrc, bgImg, cWidth, cHeight, imageBlob, action, user }) {
  const canvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const bgCanvasRef = useRef(null);
  const [selectedColor, setSelectedColor] = useState('#000000');
  const [isMouseDown, setIsMouseDown] = useState(false);
  const [undoStack, setUndoStack] = useState([]);
  const [undoRStack, setUndoRStack] = useState([]);
  const [drawingMode, setDrawingMode] = useState('draw');
  const [labelColorMap, changeLabelColorMap] = useState({});
  const [selectedMasks, setSelectedMasks] = useState([]);
  const [isFuseActive, setIsFuseActive] = useState(false); // State to track fuse mode
  const [editedBlob, setEditedBlob] = useState(null);
  const pixelSize = 1;
  const canvasWidth = cWidth;
  const canvasHeight = cHeight;

  const rgbToHex = (r, g, b) => {
    return `#${[r, g, b].map(x => x.toString(16).padStart(2, '0')).join('')}`;
  };

  const hexToRGBA = (hex) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return [r, g, b, 255]; // Full opacity
  };

  const generateUniqueColor = () => {
    let newColor;
    let newColorString;
    const existingColorStrings = new Set(Object.values(labelColorMap).map(rgb => rgb.join(',')));

    do {
      newColor = [Math.floor(Math.random() * 255), Math.floor(Math.random() * 255), Math.floor(Math.random() * 255)];
      newColorString = newColor.join(',');
    } while (existingColorStrings.has(newColorString));

    return newColor;
  };

  const updateLabelColorMap = (rgbArray) => {
    const selectedColorString = rgbArray.join(',');

    const matchingKey = Object.keys(labelColorMap).find(key =>
      labelColorMap[key].join(',') === selectedColorString
    );

    if (matchingKey) {
      console.log(`Popped color with key ${matchingKey} and value ${labelColorMap[matchingKey].join(',')}`);
      delete labelColorMap[matchingKey];
    } else {
      console.log("No matching color found to pop.");
    }

    const newColor = generateUniqueColor();
    const newKey = Object.keys(labelColorMap).length > 0 ? Math.max(...Object.keys(labelColorMap).map(Number)) + 1 : 0;
    labelColorMap[newKey] = newColor;
    changeLabelColorMap({ ...labelColorMap });

    console.log(`Added new color at index ${newKey}:`, `#${newColor.map(c => c.toString(16).padStart(2, '0')).join('')}`);
    console.log("Updated labelColorMap:", labelColorMap);

    setSelectedColor(`#${newColor.map(c => c.toString(16).padStart(2, '0')).join('')}`);
  };

  useEffect(() => {
    const bgCanvas = bgCanvasRef.current;
    const ctx = bgCanvas.getContext('2d');
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    if (bgImg) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
      };
      img.src = bgImg;
    }
  }, [bgImg]);

  useEffect(() => {
    const imageCanvas = canvasRef.current;
    const ctx = imageCanvas.getContext('2d');
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    if (imageSrc) {
      const img = new Image();
      img.onload = () => {
        const w = img.width;
        const h = img.height;
        ctx.drawImage(img, 0, 0, w, h);
      };
      img.src = imageSrc;
    }
  }, [imageSrc]);

  const changeSelectedColor = (color) => {
    setSelectedColor(color);
  };

  // Save updated mask
  const handleSave = () => {
    const imageCanvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const canvasWidth = imageCanvas.width;
    const canvasHeight = imageCanvas.height;
  
    const combinedCanvas = document.createElement('canvas');
    combinedCanvas.width = canvasWidth;
    combinedCanvas.height = canvasHeight;
    const combinedCtx = combinedCanvas.getContext('2d');
  
    combinedCtx.drawImage(imageCanvas, 0, 0);
    combinedCtx.drawImage(maskCanvas, 0, 0);
    
    const dataURL = combinedCanvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.download = "combined_image.png";
    link.href = dataURL;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  
    
    combinedCanvas.toBlob((blob) => {
      setEditedBlob(blob);
    });


  };



  //Draw
  const drawPixel = (x, y, color) => {
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas.getContext('2d');
    const rgba = hexToRGBA(color);

    ctx.fillStyle = `rgba(${rgba.join(',')})`;
    ctx.fillRect(x, y, pixelSize, pixelSize);
  };

  // Erase drawn pixels
  const clearPixel = (x, y) => {
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas.getContext('2d');
    ctx.clearRect(x, y, pixelSize, pixelSize);
  };

  //Remove original pixels - edit og mask
  const removePixel = (x, y) => {
    const maskCanvas = canvasRef.current;
    const ctx = maskCanvas.getContext('2d');
    ctx.clearRect(x, y, pixelSize, pixelSize);
  };

  // Fill mask
  const fillPixel = (startX, startY, fillColor) => {
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    const targetColor = ctx.getImageData(startX, startY, 1, 1).data;
    const fill = hexToRGBA(fillColor);

    if (fill.every((val, index) => val === targetColor[index])) {
      return;
    }

    const pixelStack = [[startX, startY]];

    while (pixelStack.length) {
      const [x, y] = pixelStack.pop();
      let currentY = y;
      let pixelPos = (currentY * canvasWidth + x) * 4;

      while (currentY >= 0 && matchColor(imageData, pixelPos, targetColor)) {
        currentY--;
        pixelPos -= canvasWidth * 4;
      }
      pixelPos += canvasWidth * 4;
      currentY++;

      let reachLeft = false;
      let reachRight = false;

      while (currentY < canvasHeight && matchColor(imageData, pixelPos, targetColor)) {
        colorPixel(imageData, pixelPos, fill);

        if (x > 0) {
          if (matchColor(imageData, pixelPos - 4, targetColor)) {
            if (!reachLeft) {
              pixelStack.push([x - 1, currentY]);
              reachLeft = true;
            }
          } else if (reachLeft) {
            reachLeft = false;
          }
        }

        if (x < canvasWidth - 1) {
          if (matchColor(imageData, pixelPos + 4, targetColor)) {
            if (!reachRight) {
              pixelStack.push([x + 1, currentY]);
              reachRight = true;
            }
          } else if (reachRight) {
            reachRight = false;
          }
        }

        currentY++;
        pixelPos += canvasWidth * 4;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  };

  //Functions for Fill
  const matchColor = (imageData, pixelPos, color) => {
    return (
      imageData.data[pixelPos] === color[0] &&
      imageData.data[pixelPos + 1] === color[1] &&
      imageData.data[pixelPos + 2] === color[2] &&
      imageData.data[pixelPos + 3] === color[3]
    );
  };

  const colorPixel = (imageData, pixelPos, color) => {
    imageData.data[pixelPos] = color[0];
    imageData.data[pixelPos + 1] = color[1];
    imageData.data[pixelPos + 2] = color[2];
    imageData.data[pixelPos + 3] = color[3];
  };

  //Undo edits
  const handleUndo = () => {
    //Undo for draw
    if (undoStack.length > 0) {
      const previousMask = undoStack.pop();
      const maskCanvas = maskCanvasRef.current;
      const ctx = maskCanvas.getContext('2d');
      ctx.putImageData(previousMask, 0, 0);
    }

    //Undo for removing original pixels
    if (undoRStack.length > 0) {
      const previousMask = undoRStack.pop();
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.putImageData(previousMask, 0, 0);
    }
  };

  // Clear canvas edits
  const handleClear = () => {
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas.getContext('2d');
    setUndoStack([...undoStack, ctx.getImageData(0, 0, canvasWidth, canvasHeight)]);

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  };

// Change mask color
  const changeColor = (x, y) => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    const tolerance = 0;
    const uniqueNewColor = generateUniqueColor()
    console.log('New mask color',uniqueNewColor)
    const fillColor = { r: uniqueNewColor[0], g: uniqueNewColor[0], b: uniqueNewColor[0], a: 255 }; 
    

    const targetColor = getPixelColor(imageData, x, y);

    const queue = [];
    const processedPixels = new Set();

    queue.push({ x, y });

    while (queue.length > 0) {
      const { x, y } = queue.shift();
      const pixelIndex = (y * canvas.width + x) * 4;

      if (processedPixels.has(pixelIndex)) {
        continue;
      }

      processedPixels.add(pixelIndex);

      const currentColor = getPixelColor(imageData, x, y);
      

      if (colorsAreSimilar(targetColor, currentColor, tolerance)) {
        setPixelColor(imageData, pixelIndex, fillColor);

        // Add neighboring pixels to the queue
        if (x > 0) queue.push({ x: x - 1, y });
        if (x < canvas.width - 1) queue.push({ x: x + 1, y });
        if (y > 0) queue.push({ x, y: y - 1 });
        if (y < canvas.height - 1) queue.push({ x, y: y + 1 });
      }
      }
      context.putImageData(imageData, 0, 0);
  };

  //Functions for change colors
  const getPixelColor = (imageData, x, y) => {
    const data = imageData.data;
    const pixelIndex = (y * imageData.width + x) * 4;
    return {
      r: data[pixelIndex],
      g: data[pixelIndex + 1],
      b: data[pixelIndex + 2],
      a: data[pixelIndex + 3],
    };
  };
  
  const setPixelColor = (imageData, pixelIndex, color) => {
    const data = imageData.data;
    data[pixelIndex] = color.r;
    data[pixelIndex + 1] = color.g;
    data[pixelIndex + 2] = color.b;
    data[pixelIndex + 3] = color.a;
  };
  
  const colorsAreSimilar = (color1, color2, tolerance) => {
    const diff =
      Math.abs(color1.r - color2.r) +
      Math.abs(color1.g - color2.g) +
      Math.abs(color1.b - color2.b) +
      Math.abs(color1.a - color2.a);
    return diff <= tolerance;
  };


  // Fuse
  const handleFuse = () => {
    const maskCanvas = maskCanvasRef.current;
    const bgCanvas = bgCanvasRef.current;
    const canvas = canvasRef.current;
    const maskCtx = maskCanvas.getContext('2d');
    const bgCtx = bgCanvas.getContext('2d');
    const canvasCtx = canvas.getContext('2d');
    const uniqueNewColor = generateUniqueColor();
    const uniqueNewHexColor = rgbToHex(...uniqueNewColor);

    // Get image data of the entire canvases
    const canvasImageData = canvasCtx.getImageData(0, 0, canvasWidth, canvasHeight);
    const canvasData = canvasImageData.data;

    const maskImageData = maskCtx.getImageData(0, 0, canvasWidth, canvasHeight);
    const maskData = maskImageData.data;

    const bgImageData = bgCtx.getImageData(0, 0, canvasWidth, canvasHeight);
    const bgData = bgImageData.data;

    const [newR, newG, newB] = uniqueNewColor;

    selectedMasks.forEach(({ color }) => {
      const [r, g, b] = hexToRGBA(color);

      for (let i = 0; i < canvasData.length; i += 4) {
        if (canvasData[i] === r && canvasData[i + 1] === g && canvasData[i + 2] === b) {
          bgData[i] = newR;
          bgData[i + 1] = newG;
          bgData[i + 2] = newB;

          maskData[i] = newR;
          maskData[i + 1] = newG;
          maskData[i + 2] = newB;
          maskData[i + 3] = 255;
        }
      }
    });

    bgCtx.putImageData(bgImageData, 0, 0);
    maskCtx.putImageData(maskImageData, 0, 0);

    setSelectedMasks([]);
    setSelectedColor(uniqueNewHexColor);
    setDrawingMode('draw'); // Reset drawing mode
    setIsFuseActive(false); // Deactivate fuse mode
    console.log("Fusing applied with new color:", uniqueNewHexColor);
  };

  useEffect(() => {
    const maskCanvas = maskCanvasRef.current;
    const canvas = canvasRef.current;
    const ctx = maskCanvas.getContext('2d');
    const canvasCtx = canvas.getContext('2d');


    const handleMouseMove = (event) => {
      if (isMouseDown) {
        const bounding = maskCanvas.getBoundingClientRect();
        const x = Math.floor((event.clientX - bounding.left) / pixelSize) * pixelSize;
        const y = Math.floor((event.clientY - bounding.top) / pixelSize) * pixelSize;

        switch (drawingMode) {
          case 'draw':
            drawPixel(x, y, selectedColor);
            break;
          case 'erase':
            clearPixel(x, y);
            break;
          case 'fuse':
            const pixelData = canvasCtx.getImageData(x, y, 1, 1).data;
            const hexColor = rgbToHex(pixelData[0], pixelData[1], pixelData[2]);
            if (hexColor !== '#000000' && !selectedMasks.some(mask => mask.color === hexColor)) {
              setSelectedMasks(prevMasks => [...prevMasks.filter(mask => mask.color !== '#000000'), { x, y, color: hexColor }]);
              console.log("Selected masks colors:", selectedMasks.map(mask => mask.color));
            }
            break;
          case 'remove':
            removePixel(x, y);
            break;
          default:
            break;
        }
      }
    };


    const handleMouseDown = (event) => {
      setIsMouseDown(true);
      const bounding = maskCanvas.getBoundingClientRect();
      const x = Math.floor((event.clientX - bounding.left) / pixelSize) * pixelSize;
      const y = Math.floor((event.clientY - bounding.top) / pixelSize) * pixelSize;

      setUndoStack([...undoStack, ctx.getImageData(0, 0, canvasWidth, canvasHeight)]);
      setUndoRStack([...undoRStack, canvasCtx.getImageData(0, 0, canvasWidth, canvasHeight)]);

      switch (drawingMode) {
        case 'fill':
          fillPixel(x, y, selectedColor);
          break;
        case 'erase':
          clearPixel(x, y);
          break;
        case 'remove':
          removePixel(x, y);
          break;
        case 'change':
          changeColor(x, y);
          break; 
        case 'fuse':
          const pixelData = canvasCtx.getImageData(x, y, 1, 1).data;
          const hexColor = rgbToHex(pixelData[0], pixelData[1], pixelData[2]);
          if (hexColor !== '#000000' && !selectedMasks.some(mask => mask.color === hexColor)) {
            setSelectedMasks(prevMasks => [...prevMasks.filter(mask => mask.color !== '#000000'), { x, y, color: hexColor }]);
            console.log("Selected masks colors:", selectedMasks.map(mask => mask.color));
          }
          break;
        default:
          drawPixel(x, y, selectedColor);
          break;
      }
    };


    const handleMouseUp = () => {
      setIsMouseDown(false);
    };

    maskCanvas.addEventListener('mousemove', handleMouseMove);
    maskCanvas.addEventListener('mousedown', handleMouseDown);
    maskCanvas.addEventListener('mouseup', handleMouseUp);

    return () => {
      maskCanvas.removeEventListener('mousemove', handleMouseMove);
      maskCanvas.removeEventListener('mousedown', handleMouseDown);
      maskCanvas.removeEventListener('mouseup', handleMouseUp);
    };
  }, [selectedColor, isMouseDown, drawingMode, undoStack, selectedMasks]);


  //Retrain
  const retrainMask = async () => {
    try {
      
      const formData = new FormData();
      formData.append('data_blob', imageBlob);
      formData.append('label_blob', editedBlob);
    
      const response = await axios.post('http://127.0.0.1:8000/retrain_model/'  + action + '/' + user + '/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
  
    } catch (error) {
      console.error('Error', error);
    }
  };
  
  return (
    <div className='row mb-5 mt-5'>
      
      <div className='col-md-12'>
      
        <p>Selected Color:
          <input
            type="color"
            value={selectedColor}
            onChange={(event) => changeSelectedColor(event.target.value)}
          />
        </p>
        <p>Current mode - {drawingMode}</p>
        {drawingMode === 'fuse' && <p>Fusing active, Colors selected: [{selectedMasks.map(mask => mask.color).join(', ')}]</p>}
        <div className='row'>
        <div className='col-md-12'>
        <button
          className={`m-1 btn btn-primary ${drawingMode === 'draw' ? 'active' : ''}`}
          onClick={() => {
            setDrawingMode('draw');
            setIsFuseActive(false);
          }}
        >
          Draw
        </button>
        <button
          className={`m-1 btn btn-primary ${drawingMode === 'fill' ? 'active' : ''}`}
          onClick={() => {
            if (drawingMode === 'fill') {
              setDrawingMode('draw');
              setIsFuseActive(false);
            } else {
              setDrawingMode('fill');
            }
          }}
        >
          Fill
        </button>
        <button
          className={`m-1 btn btn-primary ${drawingMode === 'erase' ? 'active' : ''}`}
          onClick={() => {
            if (drawingMode === 'erase') {
              setDrawingMode('draw');
              setIsFuseActive(false);
            } else {
              setDrawingMode('erase');
            }
          }}
        >
          Erase
        </button>
        </div>
        <div className='col-md-12'>
        
        <button className='m-1 btn btn-primary' onClick={handleUndo}>Undo</button>
        <button className='m-1 btn btn-primary' onClick={handleClear}>Clear</button>
        </div>
        <div className='col-md-12'>
        <button className='m-1 btn btn-primary' onClick={() => setDrawingMode('remove')}>Split cells</button>
        <button className='m-1 btn btn-primary' onClick={() => setDrawingMode('change')}>Change color</button>
        </div>
        <div className='col-md-12'>
        <button
          className={`m-1 btn btn-primary ${isFuseActive ? 'active' : ''}`}
          onClick={() => {
            if (drawingMode === 'fuse') {
              setDrawingMode('draw');
              setIsFuseActive(false);
              setSelectedMasks([]);
            } else {
              setDrawingMode('fuse');
              setIsFuseActive(true);
            }
          }}
        >
          Fuse
        </button>
        <button className='m-1 btn btn-primary' onClick={handleFuse}>Apply Fuse</button>
        </div>

        <div className='col-md-12'>
        <button className='m-1 btn btn-primary' onClick={handleSave}>Save Changes</button>
        <button className='m-1 btn btn-primary' onClick={retrainMask} >Retrain model</button>
        
        </div>
        </div>
      </div>

      <div className='col-md-12 ml-2 d-flex justify-content-center'>
        <canvas
          ref={bgCanvasRef}
          width={canvasWidth}
          height={canvasHeight}
          style={{ position: 'absolute', zIndex: 1 }}
        />
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          style={{ position: 'absolute', zIndex: 2 }}
        />
        <canvas
          ref={maskCanvasRef}
          width={canvasWidth}
          height={canvasHeight}
          style={{ position: 'absolute', zIndex: 3 }}
        />
      </div>
    </div>
  );
}

export default Canvas;
