import requests
from bs4 import BeautifulSoup

login = { # remember_web_XXX... cookie name and content

}
url = "https://deilabs.dei.unipd.it/laboratory_in_outs"
lttm_id = 20

with requests.Session() as s:
    response = s.get(url, cookies=login)
    dom = BeautifulSoup(response.text, 'html.parser')
    form = dom.find("form", {"id": "edit_laboratory_in_outs_form"})
    # check if logged in already
    if form is not None:
        data = {c.attrs["name"]: c.attrs["value"] for c in form.findChildren("input", recursive=False)}
        io = s.post(url=form.attrs["action"], json=data)
        dom = BeautifulSoup(io.text, 'html.parser')
        res = dom.find("div", {"class": "alert"}).findChildren("span", recursive=False)
        print("Logout Successful:", res[0].text)
    else:
        # we need to login
        r = input("Do you want to login in DEILabs? [Y/n]")
        if not r == "n":
            form = dom.find("form", {"id": "create_laboratory_in_out_form"})
            data = {c.attrs["name"]: c.attrs["value"] for c in form.findChildren("input", recursive=False)}
            data["laboratory_id"] = lttm_id
            io = s.post(url=form.attrs["action"], json=data)
            dom = BeautifulSoup(io.text, 'html.parser')
            res = dom.find("div", {"class": "alert"}).findChildren("span", recursive=False)
            print("Login Successful:", res[0].text)
        else:
            print("Operation cancelled by user, Goodbye.")