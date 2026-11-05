/* Offline GIF89a encoding. Quantization and LZW run in a worker; only one
   uncompressed frame is transferred at a time. No external dependencies. */
function gifEncodingWorker(){
  let palette,lut,width,height,chunks=[];
  const u16=n=>[n&255,(n>>8)&255];
  const bytes=a=>chunks.push(new Uint8Array(a));
  function train(samples){
    const hist=new Uint32Array(32768);
    for(const raw of samples){const a=new Uint8Array(raw);for(let i=0;i<a.length;i+=4)hist[((a[i]>>3)<<10)|((a[i+1]>>3)<<5)|(a[i+2]>>3)]++;}
    let bins=[];for(let k=0;k<hist.length;k++)if(hist[k])bins.push({k,n:hist[k],c:[((k>>10)&31)*8+4,((k>>5)&31)*8+4,(k&31)*8+4]});
    function box(items){const lo=[255,255,255],hi=[0,0,0];let count=0;for(const b of items){count+=b.n;for(let j=0;j<3;j++){lo[j]=Math.min(lo[j],b.c[j]);hi[j]=Math.max(hi[j],b.c[j]);}}const span=hi.map((v,j)=>v-lo[j]),axis=span.indexOf(Math.max(...span));return {items,count,axis,score:items.length>1?span[axis]*Math.sqrt(count):0};}
    const boxes=[box(bins)];while(boxes.length<256){let idx=0;for(let i=1;i<boxes.length;i++)if(boxes[i].score>boxes[idx].score)idx=i;const b=boxes[idx];if(!b.score)break;b.items.sort((a,c)=>a.c[b.axis]-c.c[b.axis]);let n=0,cut=1;for(let i=0;i<b.items.length-1;i++){n+=b.items[i].n;cut=i+1;if(n>=b.count/2)break;}boxes.splice(idx,1,box(b.items.slice(0,cut)),box(b.items.slice(cut)));}
    palette=boxes.map(b=>[0,1,2].map(j=>Math.min(255,Math.round(b.items.reduce((s,v)=>s+v.c[j]*v.n,0)/b.count))));
    // Preserve clean white backgrounds and black labels exactly.
    let white=0;for(let i=1;i<palette.length;i++)if(palette[i].reduce((s,c)=>s+(255-c)**2,0)<palette[white].reduce((s,c)=>s+(255-c)**2,0))white=i;palette[white]=[255,255,255];
    while(palette.length<256)palette.push([0,0,0]);
    lut=new Uint8Array(32768);
    for(let k=0;k<32768;k++){const c=[((k>>10)&31)*8+4,((k>>5)&31)*8+4,(k&31)*8+4];let best=0,error=Infinity;for(let j=0;j<256;j++){const p=palette[j],d=(p[0]-c[0])**2+(p[1]-c[1])**2+(p[2]-c[2])**2;if(d<error){error=d;best=j;}}lut[k]=best;}
  }
  function lzw(pixels){
    const output=[];let accum=0,bits=0,size=9,next=258,table=new Map();
    function emit(code){accum|=code<<bits;bits+=size;while(bits>=8){output.push(accum&255);accum>>>=8;bits-=8;}}
    emit(256);let prefix=pixels[0];
    for(let i=1;i<pixels.length;i++){const color=pixels[i],key=prefix*256+color,found=table.get(key);
      if(found!==undefined){prefix=found;continue;}
      // The decoder inserts one entry behind the encoder; increase the code
      // width after its previous read has completed the preceding range.
      if(next>(1<<size)&&size<12)size++;
      emit(prefix);
      if(next<4096){table.set(key,next++);}else{emit(256);table=new Map();next=258;size=9;}
      prefix=color;
    }
    if(next>(1<<size)&&size<12)size++;emit(prefix);
    if(next===(1<<size)&&size<12)size++;emit(257);if(bits)output.push(accum&255);
    return new Uint8Array(output);
  }
  self.onmessage=event=>{
    try{const d=event.data;
      if(d.type==='start'){
        width=d.width;height=d.height;chunks=[];train(d.samples);
        bytes([71,73,70,56,57,97,...u16(width),...u16(height),0xf7,0,0]);bytes(palette.flat());
        // NETSCAPE2.0 loop extension: 0 repeats indefinitely.
        bytes([0x21,0xff,11,...Array.from('NETSCAPE2.0',c=>c.charCodeAt(0)),3,1,0,0,0]);self.postMessage({type:'ready'});
      }else if(d.type==='frame'){
        const rgba=new Uint8Array(d.pixels),indexed=new Uint8Array(width*height);
        for(let i=0,j=0;i<rgba.length;i+=4,j++)indexed[j]=lut[((rgba[i]>>3)<<10)|((rgba[i+1]>>3)<<5)|(rgba[i+2]>>3)];
        const data=lzw(indexed),delay=d.delay;
        bytes([0x21,0xf9,4,4,...u16(delay),0,0,0x2c,0,0,0,0,...u16(width),...u16(height),0,8]);
        for(let i=0;i<data.length;i+=255){const block=data.subarray(i,Math.min(i+255,data.length));bytes([block.length]);chunks.push(block);}bytes([0]);self.postMessage({type:'frame'});
      }else if(d.type==='finish'){
        bytes([0x3b]);const length=chunks.reduce((s,c)=>s+c.length,0),result=new Uint8Array(length);let offset=0;for(const c of chunks){result.set(c,offset);offset+=c.length;}chunks=[];self.postMessage({type:'done',buffer:result.buffer},[result.buffer]);
      }
    }catch(error){self.postMessage({type:'error',message:String(error)});}
  };
}
