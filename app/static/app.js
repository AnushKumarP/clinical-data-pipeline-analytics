const $=id=>document.getElementById(id);
const samples=[{age:45,resting_bp:118,cholesterol:195,max_heart_rate:168,exercise_angina:0,st_depression:.2},{age:63,resting_bp:148,cholesterol:286,max_heart_rate:118,exercise_angina:1,st_depression:2.8},{age:55,resting_bp:134,cholesterol:238,max_heart_rate:145,exercise_angina:0,st_depression:1.1}];
$('sample').onclick=()=>{const s=samples[Math.floor(Math.random()*samples.length)];Object.entries(s).forEach(([k,v])=>$(k).value=v)};
fetch('/api/pipeline').then(r=>r.json()).then(d=>{$('records').textContent=d.recordsProcessed.toLocaleString();$('auc').textContent=d.rocAuc.toFixed(2)});
$('form').onsubmit=async e=>{e.preventDefault();const keys=['age','resting_bp','cholesterol','max_heart_rate','exercise_angina','st_depression'];const body=Object.fromEntries(keys.map(k=>[k,Number($(k).value)]));
 const button=e.submitter;button.disabled=true;button.firstChild.textContent='Analyzing… ';
 try{const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!response.ok)throw Error('Invalid input');const d=await response.json();
 $('empty').hidden=true;$('result').hidden=false;$('score').textContent=Math.round(d.probability*100)+'%';$('level').textContent=d.riskLevel+' indicators';$('segment').textContent=d.segment;$('cluster').textContent='#'+d.cluster;$('meter').style.width=Math.round(d.probability*100)+'%';$('factors').innerHTML=(d.topIndicators.length?d.topIndicators:['No elevated indicators']).map(x=>`<b>${x}</b>`).join('');
 }catch(err){alert('Unable to run the analysis. Please verify each value.')}finally{button.disabled=false;button.firstChild.textContent='Analyze indicators '}}
