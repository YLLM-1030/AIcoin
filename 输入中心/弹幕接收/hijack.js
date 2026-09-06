(function(){
  const B = 'http://localhost:19000/text';
  const O = window.WebSocket;
  window.WebSocket = function(){
    const w = new O(...arguments);
    w.addEventListener('message', function(e){
      try{
        var d = JSON.parse(e.data);
        if(d.cmd === 2){
          var a = d.data || [];
          var dbg = '';
          for(var i=0;i<Math.min(a.length,10);i++){
            dbg += '['+i+']='+a[i]+'|';
          }
          fetch(B, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({source:'弹幕', text: dbg})}).catch(function(){});
        }
      }catch(e){}
    });
    return w;
  };
})();
