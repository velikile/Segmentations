
window.onload = function(){}

function InferAndShowResults(inputElement)
{
    let formElement = inputElement.parentElement;
    let url = 'api/infer'
    let data = new FormData() 
    for (const pair of new FormData(formElement)) {
        data.append(pair[0], pair[1]);
    }
    fetch(url, {
        method: 'POST',
        body: data,
    }).then((res)=>(res.json())).then(res=>{
        div = document.createElement('div');
        img = document.createElement('img');
        img2 = document.createElement('img');
        img.src = res.base;
        img2.src = res.mask;
        div.append(img);
        div.append(img2);
        document.getElementsByTagName('body')[0].append(div)

    }).catch((e)=>console.log(e));

}
