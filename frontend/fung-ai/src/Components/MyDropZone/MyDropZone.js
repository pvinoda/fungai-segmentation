import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './MyDropZone.css'
import Canvas from '../../ImageComponent';
import LoadingAnimation from './LoadingAnimation';

function MyDropzone({user}) {
  const [image, setImage] = useState(null);
  const [imagePath, setImagePath] = useState(null);
  const [imageBlob, setImageBlob] = useState(null);
  const [imageBlobOriginal, setImageBlobOriginal] = useState(null);
  const [showImageUrl, setShowImageUrl] = useState('');
  const [processImageUrls, setProcessImageUrls] = useState({ flows: '', overlays: '',mask:'' });
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [imageDimensionsOriginal, setImageDimensionsOriginal] = useState({ width: 0, height: 0 });
  const [resizeFactor, setResizeFactor] = useState(1);
  const [resizedImageUrl, setResizedImageUrl] = useState('');
  const [suggestedResizeFactor,setSuggestedResizeFactor] = useState(1);
  const [labelColorMap,setLabelColorMap] = useState({});
  const [type, setType] = useState('height');
  const [rgba, setRgba] = useState([]);
  const [factor, setFactor] = useState('');
  const [sugFactor, setSugFactor] = useState('');
  const [model, setModel] = useState('proSeg');
  const [showComponent, setShowComponent] = useState(false);
  const [loading, setLoading] = useState(false);

  


  const handleCanvas = () => {
    setShowComponent(true);
  };


  const getImageDimensions = (e) => {
    const { naturalWidth, naturalHeight } = e.target;
    setImageDimensions({ width: naturalWidth, height: naturalHeight });
    setImageDimensionsOriginal({ width: naturalWidth, height: naturalHeight });
  };

// RESIZE CANVAS 
  const resizeWidth = (factor) => {
    const newWidth = imageDimensions.width * factor;
    const newHeight = imageDimensions.height ;
    setImageDimensions({ width: newWidth, height: newHeight });
    const imgElement = document.getElementById('image'); 
    
    if (imgElement) {
      imgElement.style.width = newWidth + 'px'; 
      imgElement.style.height = newHeight + 'px'; 

      const canvas = document.createElement('canvas');
      canvas.width = newWidth;
      canvas.height = newHeight;

      const ctx = canvas.getContext('2d');
      ctx.drawImage(imgElement, 0, 0, newWidth, newHeight);
      canvas.toBlob((resizedBlob) => {
        setImageBlob(resizedBlob)
      });
      console.log('resized by W',factor,'-',imageBlob);

    }
  };

// DROPZONE  
  const onDrop = useCallback((acceptedFiles) => {
    setShowImageUrl('')
    setProcessImageUrls({ flows: '', overlays: '',mask:'' })
    const file = acceptedFiles[0];
    const reader = new FileReader();

    reader.onload = async () =>  {
      const response = await fetch(reader.result);
      const blob = await response.blob();

      setImagePath(reader.result);
      setImageBlob(blob);
      setImageBlobOriginal(blob);
    };

    reader.readAsDataURL(file);
    
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });


// UPLOAD IMAGE TO SHOW IMAGE
const uploadImage = async () => {
  try {
    
    const formData = new FormData();
    formData.append('image_object', imageBlob);
    formData.append('username', user);

  
    const response = await axios.post('http://127.0.0.1:8000/show_image/' + user + '/', formData, {
      responseType: 'blob',
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    const blobData = response.data;

    console.log(blobData)
    console.log(typeof response.data)
    const url = URL.createObjectURL(blobData)
    setShowImageUrl(url)

  } catch (error) {
    console.error('Error', error);
  }
};

// SCALE FACTOR 
const scaleFactor = async () => {
  try {
    const formData = new FormData();
    formData.append('image_object', imageBlob);

    const response = await axios.post('http://127.0.0.1:8000/scale_factor/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    const f = parseFloat(response.data.scale_factor.toFixed(2));
    console.log(typeof(f))
    console.log('Scaling factor',f)
    setSugFactor(f);
    alert(`Suggested scaling factor - ${f}`);


  } catch (error) {
    console.error('Error', error);
  }
};


//  RESIZE IMAGE 
const resize = async () => {
    try {
      const formData = new FormData();
      console.log('f',factor)
      formData.append('image_object', imageBlob);
      
      const response = await axios.post('http://127.0.0.1:8000/resize_image/' + factor + '/', formData, {
      responseType: 'blob',
      headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const blobData = response.data
      console.log(blobData);
      setImageBlob(blobData);
      alert('Image resized'); 

      if(factor.charAt(0) == '-'){
        const newHeight = imageDimensions.height / parseFloat(factor)
        const newWidth = imageDimensions.width / parseFloat(factor)
        setImageDimensions({ width: newWidth, height: newHeight });
      }
      else{
        const newHeight = imageDimensions.height * parseFloat(factor)
        const newWidth = imageDimensions.width * parseFloat(factor)
        setImageDimensions({ width: newWidth, height: newHeight });
      }
      

    } catch (error) {
      console.error('Error', error);
    }
  };

// RESET IMAGE TO ORIGINAL AFTER RESIZING
  const reset = async () => {
    try {
      setImageBlob(imageBlobOriginal);
      console.log('blob reset to - ',imageBlobOriginal)
      alert('Image reset'); 
      setImageDimensions({width: imageDimensionsOriginal.width, height: imageDimensionsOriginal.height})
    } catch (error) {
      console.error('Error', error);
    }
  };

// UPLOAD IMAGE TO MODEL  
  const uploadToModel = async (action) => {
    try {
      setLoading(true);
      setProcessImageUrls({ flows: '', overlays: '',mask:'' })
      const formData = new FormData();
      formData.append('image_object', imageBlob); 
      formData.append('username', user);
  
      console.log(action)
      const processUrl = 'http://127.0.0.1:8000/process-image/' + action + '/' + user + '/';
  
      const response = await axios.post(processUrl, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const imageData = response.data;

      setProcessImageUrls({
        flows: `data:image/png;base64,${imageData.flows}`,
        overlays: `data:image/png;base64,${imageData.overlays}`,
        outlines: `data:image/png;base64,${imageData.outlines}`,
        cellprob: `data:image/png;base64,${imageData.cellprob}`,
        mask: `data:image/png;base64,${imageData.mask}`
      });
      setLabelColorMap(response.data.colors);
      console.log(response.data.colors);
      setRgba(response.data.rgba);
      console.log("Flows, Mask, and Overlays have been set to state.")
      setLoading(false);
  
    } catch (error) {
      console.error('Error', error);
    }
  };



  return (
    <div className='p-5'>
      <div className="mt-5 ">
        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''} mb-2`}>
        <input {...getInputProps()}/>
        <p>Drag and drop image here</p>
        </div>
      </div>
      <div>
      {imagePath && <button className='btn p-3 btn-primary upButton' style={{'width':'100%'}}onClick={uploadImage}>Upload Image</button>}
      </div>
    
    {showImageUrl ? (
      <>
      <div className='row mt-5'>
        <div className='d-flex justify-content-center'>
        <img onLoad={getImageDimensions} src={showImageUrl} style={{'maxHeight':'100%', 'maxWidth':'100%'}}/>
        <br/>
        </div>
        <p className='text-center'>Width: {imageDimensions.width} Height: {imageDimensions.height}</p>

        <div className='row mt-2'>
        <div className='col-md-11'><input className='form-range ' type="range" step={0.05} min={-5} max={5} value={factor} onChange={(e) => setFactor(e.target.value)}/></div>
        <p className='col-md-1 text-center'>{factor}</p>
        </div>
        <div className="row">
          <div className='col-md-4'><button className='btn btn-primary' style={{'width':'100%'}} onClick={scaleFactor}>Get suggested scaling factor</button></div>
          <div className='col-md-4'><button className='btn btn-primary' style={{'width':'100%'}}onClick={resize}>Set Dimensions for Resize</button></div>
          <div className='col-md-4'><button className='btn btn-primary' style={{'width':'100%'}}onClick={reset}>Reset Image</button></div>
        </div>



        <h4 className='pt-5'>Select model </h4>
        <div className='col-md-6 '>
        <select value={model} className='form-control' onChange={(e) => setModel(e.target.value)}>
            <option value="proSeg">proSeg</option>
            <option value="budSeg">budSeg</option>
            <option value="matSeg">matSeg</option>
            <option value="spoSeg">spoSeg</option>
            <option value="FilaBranch">FilaBranch</option>
            <option value="FilaCross_1">FilaCross_1</option>
            <option value="FilaSeg_8">FilaSeg_8</option>
            <option value="FilaSeptum_2">FilaSeptum_2</option>
            <option value="FilaTip_6">FilaTip_6</option>
            <option value="Coni_3">Coni_3</option>
        </select>
        </div>
        <button className='btn btn-primary col-md-6 modelButton' onClick={() => uploadToModel(model)}>Get prediction</button>
        <p className='text-center mt-2 fw-bold'>{loading ?  <LoadingAnimation/> : ''}</p>
      </div>
      </>
      ) : (
        <p></p>
      )}


{/* Predicted images here */}
      {processImageUrls.flows && (
        <div className='row mt-5'>
        <div className='col-md-6 mt-2'>
          <img className='processedImage ' src={processImageUrls.flows} alt="Flows Image" />
          <p className="text-center fw-bold">Flows</p>
        </div>
        <div className='col-md-6 mt-2'>
          <img className='processedImage' src={processImageUrls.overlays} alt="Overlays Image" />
          <p className="text-center fw-bold">Overlays</p>
        </div>
        <div className='col-md-6 mt-2'>
          <img className='processedImage' src={processImageUrls.outlines} alt="Outlines Image" />
          <p className="text-center fw-bold">Outlines</p>
        </div>
        <div className='col-md-6 mt-2'>
          <img className='processedImage' src={processImageUrls.cellprob} alt="Cellprob Image" />
          <p className="text-center fw-bold">Cellprob</p>
        </div>

        {!showComponent && (
        <button onClick={handleCanvas} className='btn btn-primary'>Edit mask</button>
        )}
    
        {showComponent && <Canvas imageSrc={processImageUrls.mask} bgImg={processImageUrls.overlays} labelColorMap={labelColorMap} cHeight={imageDimensions.height} cWidth={imageDimensions.width} imageBlob={imageBlob} action={model} user={user}/>}
       
      </div>
      
       
      )}

          



    </div>
  );
}

export default MyDropzone;